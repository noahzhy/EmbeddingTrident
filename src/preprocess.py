import jax
import jax.numpy as jnp


def extract_image_size(input_shape):
    if not input_shape or len(input_shape) < 4:
        return (224, 224)
    try:
        _, _, height, width = input_shape
    except ValueError:
        return (224, 224)
    if not height or not width:
        return (224, 224)
    return (int(height), int(width))


def build_preprocess_batch(image_size):

    def preprocess_one(img, pad_value=0.5):
        img = img.astype(jnp.float32) / 255.0
        height, width, channels = img.shape
        target_h, target_w = image_size
        resized = jax.image.resize(img, (target_h, target_w, channels), method="bilinear")
        scale = jnp.minimum(target_h / height, target_w / width)
        valid_h = (height * scale).astype(jnp.int32)
        valid_w = (width * scale).astype(jnp.int32)
        top = (target_h - valid_h) // 2
        left = (target_w - valid_w) // 2
        y = jnp.arange(target_h)[:, None]
        x = jnp.arange(target_w)[None, :]
        mask = ((y >= top) & (y < top + valid_h) & (x >= left) & (x < left + valid_w))[..., None]
        padded = jnp.where(mask, resized, pad_value)
        mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        padded = (padded - mean) / std
        return jnp.transpose(padded, (2, 0, 1))

    def preprocess_batch(imgs):
        return jax.vmap(preprocess_one, in_axes=(0,))(imgs)

    return jax.jit(preprocess_batch)
