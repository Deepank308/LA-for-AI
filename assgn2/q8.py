import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def valid(x, y, r, c):
    return (x >= 0 and x < r and y >= 0 and y < c)


def convolve(A, B):
    ar = A.shape[0]
    ac = A.shape[1]

    br = B.shape[0]
    bc = B.shape[1]

    cr = ar + br - 1
    cc = ac + bc - 1
    C = np.zeros((cr, cc))

    for r in range(cr):
        for s in range(cc):
            for i in range(ar):
                for j in range(ac):
                    k = r + 1 - i
                    l = s + 1 - j
            
                    if valid(k, l, br, bc):
                        C[r][s] += A[i][j]*B[k][l]

    return C


def process_image(img_path):
    img = Image.open(img_path).resize((244, 244)).convert('LA')
    return img


def get_blurring_matrices(option=0):
    matrices_list = []
    if option == 0:
        # generate mean blur matrices with p = 5, 10, 20
        p_list = [5, 10, 20]
        for p in p_list:
            matrices_list.append(np.full(shape=(p, p), fill_value=1/(p**2), dtype=np.float64))
    elif option == 1:
        matrices_list.append(np.array(np.array([[1, -1]])))

    elif option == 2:
        matrices_list.append(np.array(np.array([[1], [-1]])))
    else:
        print(f'Invalid option type. Expected: [0/1/2]. Found: {option}')

    return matrices_list


def blurr_images(img_choice, option):
    img_path = f'./dataset/{img_choice}.jpg'

    img = process_image(img_path)

    C = np.array(img)
    C = C[:, :, 0]

    B_matrices = get_blurring_matrices(option=option)

    _, axarr = plt.subplots(2, len(B_matrices))

    axarr = axarr.reshape((2, len(B_matrices)))
    for i, B in enumerate(B_matrices):
        blurred = convolve(B, C)
        axarr[0, i].imshow(img, cmap='gray', vmin = 0, vmax = 255)
        axarr[0, i].axis('off')

        axarr[1, i].imshow(blurred, cmap='gray')
        axarr[1, i].axis('off')

    print(f'Saving {img_choice}-{option}...')
    plt.savefig(f'{img_choice}-{option}.jpg')


def main():
    choices = ['cat', 'dog']
    options = [0, 1, 2]

    for choice in choices:
        for op in options:
            blurr_images(choice, op)
    
    print('Done...')


if __name__ == "__main__":
    main()
