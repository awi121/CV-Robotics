import open3d as o3d
import matplotlib.pyplot as plt

print("Read Redwood dataset")
color_raw = o3d.io.read_image("image_rgb.png")
depth_raw = o3d.io.read_image("image_depth.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)
plt.subplot(1, 2, 1)
plt.title('Rochan grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Rochan depth image')
plt.imshow(rgbd_image.depth)
plt.show()

