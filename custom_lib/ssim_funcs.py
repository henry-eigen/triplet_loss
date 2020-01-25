import numpy as np
from numpy.linalg import norm

from skimage.measure import compare_ssim as ssim


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def E(X, Y):
    return (Y - X) / norm(Y - X)

def P(X, Y):
    I = np.identity(X.size) / 255.
    E_1 = np.array([E(X, Y).flatten()])
    E_2 = np.array([E(X, Y).flatten()]).T
    return I - (E_1 * E_2)

def minimize_ssim(img, _lambda, iters=50, sigma=30):
    
    if len(img.shape) != 3:
        img = img[0]
    
    noise = np.random.randint(0, 255, size = img.shape) / 255.
    noise = noise * _lambda / norm(noise)
    adv_x = np.clip(img + noise, 0, 1)
    
    prev_ssim = ssim(img, adv_x, multichannel=True)
    prev_img = np.copy(adv_x)
    
    for i in range(iters):

        #      ----------------- step 1 -----------------
        grad = ssim(img, adv_x, multichannel=True, gradient=True)[1].flatten()
        grad = np.expand_dims(grad, axis=1) 

        diff = E(img, adv_x).flatten()
        diff = np.expand_dims(diff, axis=1) 

        diff_t = diff.T

        update = sigma * (grad - diff.dot( (diff_t.dot(grad)) ))
        update = update.reshape(adv_x.shape)

        adv_x = adv_x - update

        #      ----------------- step 2 -----------------
        adv_x = np.clip(img + _lambda * E(img, adv_x), 0, 1)


        # ----------- Conditional update -----------
        ssim_score = ssim(img, adv_x, multichannel=True)

        if ssim_score > prev_ssim:
            adv_x = np.copy(prev_img)
            sigma /= 2
        else:
            prev_img = np.copy(adv_x)
            prev_ssim = ssim_score
            
        

    return adv_x

def maximize_ssim(img, _lambda, iters=50, sigma=30):
    
    if len(img.shape) != 3:
        img = img[0]
    
    noise = np.random.randint(0, 255, size = img.shape) / 255.
    noise = noise * _lambda / norm(noise)
    adv_x = np.clip(img + noise, 0, 1)
        
    prev_ssim = ssim(img, adv_x, multichannel=True)
    prev_img = np.copy(adv_x)

    for i in range(iters):

        #      ----------------- step 1 -----------------
        grad = ssim(img, adv_x, multichannel=True, gradient=True)[1].flatten()
        grad = np.expand_dims(grad, axis=1) 

        diff = E(img, adv_x).flatten()
        diff = np.expand_dims(diff, axis=1) 

        diff_t = diff.T

        update = sigma * (grad - diff.dot( (diff_t.dot(grad)) ))
        update = update.reshape(adv_x.shape)

        adv_x = adv_x + update

        #      ----------------- step 2 -----------------
        adv_x = np.clip(img + _lambda * E(img, adv_x), 0, 1)


        # ----------- Conditional update -----------
        ssim_score = ssim(img, adv_x, multichannel=True)

        if ssim_score < prev_ssim:
            adv_x = np.copy(prev_img)
            sigma /= 2
        else:
            prev_img = np.copy(adv_x)
            prev_ssim = ssim_score
            
    return adv_x