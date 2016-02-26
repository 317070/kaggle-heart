import numpy as np
import numpy as np
import skimage.io
import skimage.transform
from image_transform import perturb
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.pyplot as plt
from skimage import io

image = np.clip(io.imread("dickbutt.jpg"),0.0, 1.0)[:,:,0]
print image.shape


result = perturb(image, target_shape=(500, 500), augmentation_params={"zoom_range":[0.05, 0.05],
                                             "rotation_range":[0.0, 0.0],
                                             "shear_range":[0, 0],
                                             "skew_x_range":[0, 0],
                                             "skew_y_range":[0, 0],
                                             "translation_range":[0.0, 0.0],
                                             "do_flip":False,
                                             "allow_stretch":False})


fig = plt.figure()
mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(50, 100, 600, 300)

im1 = fig.gca().imshow(result, cmap='gist_gray_r', vmin=0, vmax=1)

def init():
    im1.set_data(result)

def animate(i):
    result = perturb(image, target_shape=(500, 500),
                     augmentation_params={"rotation_range":[float(i), float(i)],
                                          "zoom_range":[0.5, 0.5],
                                          "skew_x_range":[-20, 20],
                                          "skew_y_range":[-20, 20],
                                          "do_flip":False,
                                          "allow_stretch":True})
    fig.suptitle("shear %f"%float(i))
    im1.set_data(result)
    return im1

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=50)
#anim.save('my_animation.mp4')
plt.show()