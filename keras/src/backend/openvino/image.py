import itertools
import math

import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output

MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}

SCALE_AND_TRANSLATE_METHODS = {
    "linear",
    "bilinear",
    "trilinear",
    "cubic",
    "bicubic",
    "tricubic",
    "lanczos3",
    "lanczos5",
}


def _ov_fill_triangle_kernel(x, ov_type):
    return ov_opset.maximum(
        ov_opset.constant(0.0, ov_type),
        ov_opset.subtract(ov_opset.constant(1.0, ov_type), ov_opset.abs(x)),
    ).output(0)


def _ov_fill_keys_cubic_kernel(x, ov_type):
    def c(v):
        return ov_opset.constant(v, ov_type).output(0)

    out = ov_opset.add(
        ov_opset.multiply(
            ov_opset.subtract(
                ov_opset.multiply(c(1.5), x).output(0), c(2.5)
            ).output(0),
            ov_opset.multiply(x, x).output(0),
        ).output(0),
        c(1.0),
    ).output(0)
    out2 = ov_opset.add(
        ov_opset.multiply(
            ov_opset.subtract(
                ov_opset.multiply(
                    ov_opset.add(
                        ov_opset.multiply(c(-0.5), x).output(0), c(2.5)
                    ).output(0),
                    x,
                ).output(0),
                c(4.0),
            ).output(0),
            x,
        ).output(0),
        c(2.0),
    ).output(0)
    out = ov_opset.select(
        ov_opset.greater_equal(x, c(1.0)).output(0), out2, out
    ).output(0)
    return ov_opset.select(
        ov_opset.greater_equal(x, c(2.0)).output(0), c(0.0), out
    ).output(0)


def _ov_fill_lanczos_kernel(radius, x, ov_type):
    def c(v):
        return ov_opset.constant(v, ov_type).output(0)

    pi_x = ov_opset.multiply(c(math.pi), x).output(0)
    pi_x_r = ov_opset.multiply(c(math.pi / float(radius)), x).output(0)
    y = ov_opset.multiply(
        ov_opset.multiply(
            c(float(radius)), ov_opset.sin(pi_x).output(0)
        ).output(0),
        ov_opset.sin(pi_x_r).output(0),
    ).output(0)
    x2 = ov_opset.multiply(x, x).output(0)
    denom = ov_opset.multiply(c(math.pi**2), x2).output(0)
    safe_denom = ov_opset.select(
        ov_opset.not_equal(x, c(0.0)).output(0), denom, c(1.0)
    ).output(0)
    out = ov_opset.select(
        ov_opset.greater(x, c(1e-3)).output(0),
        ov_opset.divide(y, safe_denom).output(0),
        c(1.0),
    ).output(0)
    return ov_opset.select(
        ov_opset.greater(x, c(float(radius))).output(0), c(0.0), out
    ).output(0)


_ov_kernels = {
    "linear": _ov_fill_triangle_kernel,
    "cubic": _ov_fill_keys_cubic_kernel,
    "lanczos3": lambda x, t: _ov_fill_lanczos_kernel(3.0, x, t),
    "lanczos5": lambda x, t: _ov_fill_lanczos_kernel(5.0, x, t),
}


def _ov_compute_weight_mat(
    input_size, output_size, scale_i, translation_i, kernel, antialias, ov_type
):
    """Compute (input_size, output_size) resampling weight matrix."""

    def c(v):
        return ov_opset.constant(v, ov_type).output(0)

    inv_scale = ov_opset.divide(c(1.0), scale_i).output(0)
    kernel_scale = (
        ov_opset.maximum(inv_scale, c(1.0)).output(0) if antialias else c(1.0)
    )

    out_range = ov_opset.range(
        c(0.0),
        c(float(output_size)),
        c(1.0),
        ov_type.get_type_name(),
    ).output(0)
    sample_f = ov_opset.subtract(
        ov_opset.subtract(
            ov_opset.multiply(
                ov_opset.add(out_range, c(0.5)).output(0), inv_scale
            ).output(0),
            ov_opset.multiply(translation_i, inv_scale).output(0),
        ).output(0),
        c(0.5),
    ).output(0)

    in_range = ov_opset.range(
        c(0.0),
        c(float(input_size)),
        c(1.0),
        ov_type.get_type_name(),
    ).output(0)

    sf2d = ov_opset.unsqueeze(sample_f, ov_opset.constant(0, Type.i64)).output(
        0
    )
    ir2d = ov_opset.unsqueeze(in_range, ov_opset.constant(1, Type.i64)).output(
        0
    )

    x = ov_opset.divide(
        ov_opset.abs(ov_opset.subtract(sf2d, ir2d).output(0)).output(0),
        kernel_scale,
    ).output(0)

    weights = kernel(x, ov_type)

    total = ov_opset.reduce_sum(
        weights, ov_opset.constant([0], Type.i32), keep_dims=True
    ).output(0)

    eps = 1000.0 * float(np.finfo(np.float32).eps)
    safe_total = ov_opset.select(
        ov_opset.not_equal(total, c(0.0)).output(0), total, c(1.0)
    ).output(0)
    weights = ov_opset.select(
        ov_opset.greater(ov_opset.abs(total).output(0), c(eps)).output(0),
        ov_opset.divide(weights, safe_total).output(0),
        c(0.0),
    ).output(0)

    valid = ov_opset.logical_and(
        ov_opset.greater_equal(sf2d, c(-0.5)).output(0),
        ov_opset.less_equal(sf2d, c(float(input_size) - 0.5)).output(0),
    ).output(0)
    return ov_opset.select(valid, weights, c(0.0)).output(0)


def _ov_contract_dim(x, w, d):
    """Equivalent to tensordot(x, w, axes=(d, 0)) then moveaxis(-1, d)."""
    N = x.get_partial_shape().rank.get_length()
    perm_fwd = list(range(d)) + list(range(d + 1, N)) + [d]
    if perm_fwd != list(range(N)):
        x = ov_opset.transpose(x, ov_opset.constant(perm_fwd, Type.i64)).output(
            0
        )
    x = ov_opset.unsqueeze(x, ov_opset.constant(N - 1, Type.i64)).output(0)
    x = ov_opset.matmul(x, w, False, False).output(0)
    x = ov_opset.squeeze(x, ov_opset.constant([N - 1], Type.i64)).output(0)
    perm_back = list(range(d)) + [N - 1] + list(range(d, N - 1))
    if perm_back != list(range(N)):
        x = ov_opset.transpose(
            x, ov_opset.constant(perm_back, Type.i64)
        ).output(0)
    return x


def rgb_to_grayscale(images, data_format=None):
    images = get_ov_output(images)
    data_format = backend.standardize_data_format(data_format)
    if images.get_partial_shape().rank not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    channel_axis = -3 if data_format == "channels_first" else -1
    if images.shape[channel_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )

    if images.shape[channel_axis] == 3:
        original_type = images.get_element_type()
        rgb_weights = ov_opset.constant(
            [0.2989, 0.5870, 0.1140], dtype=original_type
        ).output(0)
        if data_format == "channels_first":
            rgb_weights = ov_opset.unsqueeze(rgb_weights, axes=[-2, -1]).output(
                0
            )
        grayscales = ov_opset.multiply(images, rgb_weights).output(0)
        grayscales = ov_opset.reduce_sum(
            grayscales, reduction_axes=[channel_axis]
        ).output(0)
        grayscales = ov_opset.unsqueeze(grayscales, axes=[channel_axis]).output(
            0
        )
        if grayscales.get_element_type() != original_type:
            # Type of grayscales may be changed after unsqueeze, so we need to
            # convert it back to the original type.
            grayscales = ov_opset.convert(grayscales, original_type).output(0)

    return OpenVINOKerasTensor(grayscales)


def rgb_to_hsv(images, data_format=None):
    dtype = images.dtype
    images = get_ov_output(images)
    ov_type = images.get_element_type()
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    eps = ov_opset.constant(backend.epsilon(), dtype=ov_type).output(0)
    images = ov_opset.select(
        ov_opset.less(ov_opset.abs(images), eps),
        ov_opset.constant(0.0, dtype=ov_type),
        images,
    ).output(0)
    rgb_channels = ov_opset.split(images, axis=channels_axis, num_splits=3)
    r, g, b = (
        rgb_channels.output(0),
        rgb_channels.output(1),
        rgb_channels.output(2),
    )

    def rgb_planes_to_hsv_planes(r, g, b):
        value = ov_opset.maximum(ov_opset.maximum(r, g), b).output(0)
        minimum = ov_opset.minimum(ov_opset.minimum(r, g), b).output(0)
        range_ = ov_opset.subtract(value, minimum).output(0)

        safe_value = ov_opset.select(
            ov_opset.greater(value, ov_opset.constant(0.0, dtype=ov_type)),
            value,
            ov_opset.constant(1.0, dtype=ov_type),
        ).output(0)
        safe_range = ov_opset.select(
            ov_opset.greater(range_, ov_opset.constant(0.0, dtype=ov_type)),
            range_,
            ov_opset.constant(1.0, dtype=ov_type),
        ).output(0)

        saturation = ov_opset.select(
            ov_opset.greater(value, ov_opset.constant(0.0, dtype=ov_type)),
            ov_opset.divide(range_, safe_value),
            ov_opset.constant(0.0, dtype=ov_type),
        ).output(0)
        norm = ov_opset.divide(
            ov_opset.constant(1.0, dtype=ov_type),
            ov_opset.multiply(
                ov_opset.constant(6.0, dtype=ov_type), safe_range
            ),
        ).output(0)

        hue = ov_opset.select(
            ov_opset.equal(value, g),
            ov_opset.add(
                ov_opset.multiply(norm, ov_opset.subtract(b, r)),
                ov_opset.constant(2.0 / 6.0, dtype=ov_type),
            ),
            ov_opset.add(
                ov_opset.multiply(norm, ov_opset.subtract(r, g)),
                ov_opset.constant(4.0 / 6.0, dtype=ov_type),
            ),
        ).output(0)
        hue = ov_opset.select(
            ov_opset.equal(value, r),
            ov_opset.multiply(norm, ov_opset.subtract(g, b)),
            hue,
        ).output(0)
        hue = ov_opset.select(
            ov_opset.greater(range_, ov_opset.constant(0.0, dtype=ov_type)),
            hue,
            ov_opset.constant(0.0, dtype=ov_type),
        ).output(0)
        hue = ov_opset.add(
            hue,
            ov_opset.convert(
                ov_opset.less(hue, ov_opset.constant(0.0, dtype=ov_type)),
                ov_type,
            ),
        ).output(0)
        return hue, saturation, value

    images = ov_opset.concat(
        rgb_planes_to_hsv_planes(r, g, b), axis=channels_axis
    ).output(0)
    return OpenVINOKerasTensor(images)


def hsv_to_rgb(images, data_format=None):
    dtype = images.dtype
    images = get_ov_output(images)
    ov_type = images.get_element_type()
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    hsv_channels = ov_opset.split(images, axis=channels_axis, num_splits=3)
    hue, saturation, value = (
        hsv_channels.output(0),
        hsv_channels.output(1),
        hsv_channels.output(2),
    )

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        def channel_value(channel_delta, one_minus_saturation):
            return ov_opset.multiply(
                value,
                ov_opset.add(
                    one_minus_saturation,
                    ov_opset.multiply(saturation, channel_delta),
                ),
            )

        dh = ov_opset.multiply(
            ov_opset.mod(hue, ov_opset.constant(1.0, dtype=ov_type)),
            ov_opset.constant(6.0, dtype=ov_type),
        ).output(0)
        one_const = ov_opset.constant(1.0, dtype=ov_type).output(0)
        two_const = ov_opset.constant(2.0, dtype=ov_type).output(0)
        three_const = ov_opset.constant(3.0, dtype=ov_type).output(0)
        four_const = ov_opset.constant(4.0, dtype=ov_type).output(0)
        dr = ov_opset.subtract(
            ov_opset.abs(ov_opset.subtract(dh, three_const)), one_const
        ).output(0)
        dr = ov_opset.clamp(dr, 0.0, 1.0).output(0)
        dg = ov_opset.subtract(
            two_const, ov_opset.abs(ov_opset.subtract(dh, two_const))
        ).output(0)
        dg = ov_opset.clamp(dg, 0.0, 1.0).output(0)
        db = ov_opset.subtract(
            two_const, ov_opset.abs(ov_opset.subtract(dh, four_const))
        ).output(0)
        db = ov_opset.clamp(db, 0.0, 1.0).output(0)
        one_minus_saturation = ov_opset.subtract(one_const, saturation).output(
            0
        )

        red = channel_value(dr, one_minus_saturation)
        green = channel_value(dg, one_minus_saturation)
        blue = channel_value(db, one_minus_saturation)
        return red, green, blue

    images = ov_opset.concat(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    ).output(0)
    return OpenVINOKerasTensor(images)


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format="channels_last",
):
    raise NotImplementedError("`resize` is not supported with openvino backend")


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`affine_transform` is not supported with openvino backend"
    )


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`perspective_transform` is not supported with openvino backend"
    )


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    if fill_mode not in MAP_COORDINATES_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{MAP_COORDINATES_FILL_MODES}. Received: fill_mode={fill_mode}"
        )
    if order not in (0, 1):
        raise ValueError(
            "Invalid value for argument `order`. Expected one of "
            f"{[0, 1]}. Received: order={order}"
        )

    inputs_ov = get_ov_output(inputs)
    ov_type = inputs_ov.get_element_type()
    N = inputs_ov.get_partial_shape().rank.get_length()
    input_shape = [int(d) for d in inputs_ov.shape]

    if isinstance(coordinates, (list, tuple)):
        if len(coordinates) != N:
            raise ValueError(
                "First dim of `coordinates` must be the same as the rank of "
                "`inputs`. "
                f"Received inputs with shape: {inputs_ov.shape} and coordinate "
                f"leading dim of {len(coordinates)}"
            )
        coord_arrs = [get_ov_output(c) for c in coordinates]
    else:
        coord_ov = get_ov_output(coordinates)
        if len(coord_ov.shape) < 2:
            raise ValueError(
                "Invalid coordinates rank: expected at least rank 2."
                f" Received input with shape: {coord_ov.shape}"
            )
        if coord_ov.shape[0] != N:
            raise ValueError(
                "First dim of `coordinates` must be the same as the rank of "
                "`inputs`. "
                f"Received inputs with shape: {inputs_ov.shape} and coordinate "
                f"leading dim of {coord_ov.shape[0]}"
            )
        coord_arrs = [
            ov_opset.gather(
                coord_ov,
                ov_opset.constant(i, Type.i64),
                ov_opset.constant(0, Type.i64),
            ).output(0)
            for i in range(N)
        ]

    coord_ov_type = coord_arrs[0].get_element_type()

    strides = []
    s = 1
    for sz in reversed(input_shape):
        strides.insert(0, s)
        s *= sz

    flat_input = ov_opset.reshape(
        inputs_ov, ov_opset.constant([-1], Type.i64), True
    ).output(0)

    coord_arrs_flat = [
        ov_opset.reshape(c, ov_opset.constant([-1], Type.i64), True).output(0)
        for c in coord_arrs
    ]

    def get_interp_nodes(c):
        if order == 0:
            idx = ov_opset.convert(ov_opset.round(c).output(0), "i32").output(0)
            return [(idx, None)]
        lower = ov_opset.floor(c).output(0)
        upper_w = ov_opset.subtract(c, lower).output(0)
        lower_w = ov_opset.subtract(
            ov_opset.constant(1.0, coord_ov_type).output(0), upper_w
        ).output(0)
        idx = ov_opset.convert(lower, "i32").output(0)
        idx1 = ov_opset.add(idx, ov_opset.constant(1, Type.i32)).output(0)
        return [(idx, lower_w), (idx1, upper_w)]

    def process_coord(idx, size):
        """Return (safe_idx_i64, valid_mask_or_None)."""
        if fill_mode == "constant":
            valid = ov_opset.logical_and(
                ov_opset.greater_equal(
                    idx, ov_opset.constant(0, Type.i32)
                ).output(0),
                ov_opset.less(idx, ov_opset.constant(size, Type.i32)).output(0),
            ).output(0)
            safe = ov_opset.clamp(idx, 0, size - 1).output(0)
        elif fill_mode == "nearest":
            safe = ov_opset.clamp(idx, 0, size - 1).output(0)
            valid = None
        elif fill_mode == "wrap":
            size_c = ov_opset.constant(size, Type.i32).output(0)
            safe = ov_opset.floor_mod(idx, size_c).output(0)
            valid = None
        else:  # mirror / reflect
            size_c = ov_opset.constant(size, Type.i32).output(0)
            size_2c = ov_opset.constant(size * 2, Type.i32).output(0)
            abs_idx = ov_opset.abs(idx).output(0)
            mod = ov_opset.floor_mod(abs_idx, size_2c).output(0)
            under = ov_opset.less(mod, size_c).output(0)
            safe = ov_opset.select(
                under,
                mod,
                ov_opset.subtract(size_2c, mod).output(0),
            ).output(0)
            if fill_mode == "reflect":
                safe = ov_opset.select(
                    ov_opset.logical_not(under).output(0),
                    ov_opset.subtract(
                        safe, ov_opset.constant(1, Type.i32).output(0)
                    ).output(0),
                    safe,
                ).output(0)
            valid = None
        return ov_opset.convert(safe, "i64").output(0), valid

    per_dim_nodes = []
    for i in range(N):
        nodes = []
        for idx, weight in get_interp_nodes(coord_arrs_flat[i]):
            safe_i64, valid = process_coord(idx, input_shape[i])
            nodes.append((safe_i64, valid, weight))
        per_dim_nodes.append(nodes)

    is_int = ov_type not in (Type.f16, Type.bf16, Type.f32, Type.f64)
    output_ov = None
    for items in itertools.product(*per_dim_nodes):
        indices, validities, weights = zip(*items)

        flat_idx = None
        for idx_i64, stride in zip(indices, strides):
            contrib = ov_opset.multiply(
                idx_i64,
                ov_opset.constant(stride, Type.i64).output(0),
            ).output(0)
            flat_idx = (
                contrib
                if flat_idx is None
                else ov_opset.add(flat_idx, contrib).output(0)
            )

        gathered = ov_opset.gather(
            flat_input, flat_idx, ov_opset.constant(0, Type.i64)
        ).output(0)
        gathered = ov_opset.convert(gathered, coord_ov_type).output(0)

        if fill_mode == "constant":
            all_valid = validities[0]
            for v in validities[1:]:
                all_valid = ov_opset.logical_and(all_valid, v).output(0)
            fill_c = ov_opset.constant(float(fill_value), coord_ov_type).output(
                0
            )
            gathered = ov_opset.select(all_valid, gathered, fill_c).output(0)

        if order == 0:
            output_ov = gathered
        else:
            combined_w = weights[0]
            for w in weights[1:]:
                combined_w = ov_opset.multiply(combined_w, w).output(0)
            contribution = ov_opset.multiply(gathered, combined_w).output(0)
            output_ov = (
                contribution
                if output_ov is None
                else ov_opset.add(output_ov, contribution).output(0)
            )

    if is_int:
        output_ov = ov_opset.round(output_ov).output(0)
    output_ov = ov_opset.convert(output_ov, ov_type).output(0)

    out_shape = ov_opset.convert(
        ov_opset.shape_of(coord_arrs[0]).output(0), "i64"
    ).output(0)
    output_ov = ov_opset.reshape(output_ov, out_shape, True).output(0)
    return OpenVINOKerasTensor(output_ov)


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    raise NotImplementedError(
        "`gaussian_blur` is not supported with openvino backend"
    )


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    raise NotImplementedError(
        "`elastic_transform` is not supported with openvino backend"
    )


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    if method not in SCALE_AND_TRANSLATE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{SCALE_AND_TRANSLATE_METHODS}. Received: method={method}"
        )
    if method in ("bilinear", "trilinear"):
        method = "linear"
    elif method in ("bicubic", "tricubic"):
        method = "cubic"

    images_ov = get_ov_output(images)
    scale_ov = get_ov_output(scale)
    translation_ov = get_ov_output(translation)
    ov_type = images_ov.get_element_type()
    N = images_ov.get_partial_shape().rank.get_length()

    kernel = _ov_kernels[method]
    is_int = ov_type not in (Type.f16, Type.bf16, Type.f32, Type.f64)
    if is_int:
        compute_type = Type.f32
        output = ov_opset.convert(images_ov, compute_type).output(0)
    else:
        compute_type = ov_type
        output = images_ov

    scale_ov = ov_opset.convert(scale_ov, compute_type).output(0)
    translation_ov = ov_opset.convert(translation_ov, compute_type).output(0)

    for i, d in enumerate(spatial_dims):
        d = d % N
        input_size = int(images_ov.shape[d])
        output_size = int(output_shape[d])

        scale_i = ov_opset.gather(
            scale_ov,
            ov_opset.constant(i, Type.i64),
            ov_opset.constant(0, Type.i64),
        ).output(0)
        translation_i = ov_opset.gather(
            translation_ov,
            ov_opset.constant(i, Type.i64),
            ov_opset.constant(0, Type.i64),
        ).output(0)

        w = _ov_compute_weight_mat(
            input_size,
            output_size,
            scale_i,
            translation_i,
            kernel,
            antialias,
            compute_type,
        )
        output = _ov_contract_dim(output, w, d)

    if is_int:
        all_axes = list(range(N))
        min_val = ov_opset.convert(
            ov_opset.reduce_min(
                images_ov,
                ov_opset.constant(all_axes, Type.i32),
                keep_dims=False,
            ).output(0),
            compute_type,
        ).output(0)
        max_val = ov_opset.convert(
            ov_opset.reduce_max(
                images_ov,
                ov_opset.constant(all_axes, Type.i32),
                keep_dims=False,
            ).output(0),
            compute_type,
        ).output(0)
        output = ov_opset.maximum(
            ov_opset.round(output).output(0), min_val
        ).output(0)
        output = ov_opset.minimum(output, max_val).output(0)
        output = ov_opset.convert(output, ov_type).output(0)

    return OpenVINOKerasTensor(output)
