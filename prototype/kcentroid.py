# k-centroid downscaling (Astropulse, MIT — from the pixeldetector project).
# Per output pixel: k-means the source tile to `centroids` colours, keep the
# dominant one. Retains outlines that NEAREST's single-sample destroys.
from PIL import Image


def k_centroid(image: Image.Image, width: int, height: int, centroids: int = 2) -> Image.Image:
    image = image.convert("RGB")
    out = Image.new("RGB", (width, height))
    wf = image.width / width
    hf = image.height / height
    for x in range(width):
        for y in range(height):
            tile = image.crop((int(x * wf), int(y * hf), int((x + 1) * wf), int((y + 1) * hf)))
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids)
            counts = tile.getcolors()
            idx = max(counts, key=lambda c: c[0])[1]
            pal = tile.getpalette()
            out.putpixel((x, y), tuple(pal[idx * 3:idx * 3 + 3]))
    return out
