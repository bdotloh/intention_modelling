import cv2
import os
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np

sprite_map = cv2.imread('Spritemap.png')
urgent_bed_map = cv2.imread('Urgency_bed_up.png')
#24 pixels x 24 pixelsa
urgent_bed_up = urgent_bed_map[0:48, 0:48]
bed_up = sprite_map[0:24, 0:24] #yellow
bed_down = sprite_map[0:24, 24:48] #yellow
toilet = sprite_map[0:24, 48:72]
nurse = sprite_map[0:24, 72:96] #red
nurse_station = sprite_map[0:24, 96:120] #blue
wall = sprite_map[0:24, 120:144]



def save_img(rgb_arr, path, name):
    plt.imshow(rgb_arr, interpolation='nearest')
    plt.savefig(path + name)


def make_video_from_image_dir(vid_path, img_folder, video_name='trajectory', fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)



def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                             fps=5, format="mp4v", resize=(288, 384)):  #264, 312
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")

        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        image = change_to_sprite(image)
        video.write(image)

    video.release()
    cv2.destroyAllWindows()

def change_to_sprite(image):
    new_image=np.zeros([384,288,3], dtype=np.uint8)
    for row_idx in range(int(len(image)/24)):
        for col_idx in range(int(len(image[0])/24)):
            row=row_idx*24+5
            col=col_idx*24+5
            pixel = image[row][col]
            up_pixel = image[row-24][col]

            # 'B': [255, 222, 173],  # Beds
            # if yellow, change to bed_up
            if pixel[2]==255 and pixel[1]==222 and up_pixel[2]==105 and up_pixel[1]==105:
                for pixel_row in range(len(bed_up)):
                    for pixel_col in range(len(bed_up[0])):
                        new_image[row-5 +pixel_row][col-5 +pixel_col] = bed_up[pixel_row][pixel_col]
            # if yellow, change to bed_down
            elif pixel[2]==255 and pixel[1]==222 and up_pixel[2]==0 and up_pixel[1]==0:
                for pixel_row in range(len(bed_down)):
                    for pixel_col in range(len(bed_down[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = bed_down[pixel_row][pixel_col]

            # if blue, change to urgent_bed_up
            elif pixel[2] == 2 and pixel[1] == 81:
                for pixel_row in range(len(toilet)):
                    for pixel_col in range(len(toilet[0])):
                        new_image[row-5+pixel_row][col-5+pixel_col] = urgent_bed_up[pixel_row][pixel_col]

            # if purple, change to toilet
            elif pixel[2]==188 and pixel[1]==143:
                for pixel_row in range(len(toilet)):
                    for pixel_col in range(len(toilet[0])):
                        new_image[row-5+pixel_row][col-5+pixel_col] = toilet[pixel_row][pixel_col]
            # if red, change to nurse
            elif pixel[2]==205 and pixel[1]==92:
                for pixel_row in range(len(nurse)):
                    for pixel_col in range(len(nurse[0])):
                        new_image[row-5+pixel_row][col-5+pixel_col] = nurse[pixel_row][pixel_col]
            # if cyan, change to nurse_station
            elif pixel[2]==95 and pixel[1]==158:
                for pixel_row in range(len(nurse_station)):
                    for pixel_col in range(len(nurse_station[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = nurse_station[pixel_row][pixel_col]
            elif pixel[2] == 105 and pixel[1] == 105:
                for pixel_row in range(len(wall)):
                    for pixel_col in range(len(wall[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = wall[pixel_row][pixel_col]

    return new_image


def return_view(grid, pos, row_size, col_size):
    """Given a map env, position and view window, returns correct map part
    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros
    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: list
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension
    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge,
                                               top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size: x + col_size + 1,
                   y - row_size: y + row_size + 1]
    return view


def pad_if_needed(left_edge, right_edge, top_edge, bot_edge, matrix):
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, top_pad


def pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
    pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                     'constant', constant_values=(const_val, const_val))
    return pad_mat