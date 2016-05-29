#encoding: utf-8

import sys
import pygame
import os
from pygame.locals import *
from pygame import freetype
from PIL import Image
import numpy as np
from numpy.random import rand
from numpy import float32, uint8, int32
from skimage.io import imshow, show, imsave
from skimage.transform import rescale, rotate, resize
from skimage.color import rgb2grey

from progressbar import ProgressBar

import cPickle

import cv2

white = (255, 255, 255)
black = (0, 0, 0)
red = (188, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

def save_screen_img(pg_surface, fn, quality=100):
    imgstr = pygame.image.tostring(pg_surface, 'RGB')
    im = Image.fromstring('RGB', pg_surface.get_size(), imgstr)
    im.save(fn, quality=quality)
    print fn

def toNumpyArr(pg_surface):
    imgstr = pygame.image.tostring(pg_surface, 'RGB')
    im = Image.fromstring('RGB', pg_surface.get_size(), imgstr)
    npArr = np.asarray(im)
    npArr.flags.writeable = True
    return npArr

def load_fonts(dir='/Volumes/export/Fonts/font_all'):
    files = []
    for f in os.listdir(dir):
        if '.ttf' in f:
            files.append(os.path.join(dir, f))

    return files

def load_chars(file='./char.dat'):
    with open(file, 'r') as f:
        chars = f.readlines()
    return chars

def rotation(image, font_rotate):
    image = np.ones(image.shape) - image
    image = rotate(image, font_rotate)
    image = np.ones(image.shape) - image
    return image

def low_resolusion(image, sz):
    image = rescale(image, np.random.rand() / 2.0 + 0.5)
    image = resize(image, sz)
    return image

def expand(image, sz):
    image = rescale(image, [1.0, rand() * 0.3 + 1.0])
    h, w, c = image.shape
    ch, cw = int32(h / 2), int32(w / 2)
    if ch > cw:
        image = image[ch-cw:ch+cw, 0:w, :]
    else:
        image = image[0:h, cw-ch:cw+ch, :]

    image = resize(image, sz)
    return image


#global gameDisplay

if __name__ == '__main__':
    sz = (48, 48)
    """ 保存先 """
    prefix = '../data/pkl'
    pygame.init()
    #gameDisplay = pygame.display.set_mode(sz)
    #pygame.display.set_caption('test')
    fontfiles = load_fonts()
    chars = load_chars()

    """ バックグラウンド サーフェスを生成 """
    bg_surf = pygame.Surface(sz, SRCALPHA, 32)

    """ 文字とidのセットを書き出します """
    f = open('label.txt', 'w')

    progress = ProgressBar()
    progress.min_value = 0
    progress.max_value = len(chars)
    
    for id, c in enumerate(chars):
        progress.update(id+1)
        labels = []
        data = []
        pklfile = os.path.join(prefix,'data_{}.pkl'.format(id))
        text = c.decode('utf-8').rstrip()

        # 各文字何枚生成するか
        for i in range(500):
            bg_surf.fill(white)
            """ 描画領域設定 サーフェスはフレーム中央に配置 """
            bg_rect = bg_surf.get_rect()
            bg_rect.center = (sz[0] / 2, sz[1] / 2)

            """ フォント　サーフェス作成からレンダリングまで """
            from numpy.random import rand
            font_size = np.int(rand()*20 + 40) # サイズに注意
            fontname = np.random.choice(fontfiles)
            #print fontname
            font_rotate = np.int(rand()*90 - 45)
        
            #print fontname
            font = freetype.Font(fontname, font_size)
            font.underline = False
            rect = font.get_rect(text)
            rect.center = (sz[0] / 2, sz[1] / 2)
            font.render_to(bg_surf, rect, text)
            #pygame.PixelArray(bg_surf).replace((0, 0, 0), (0, 255, 0), distance=0.7)

            """ sequential step """
            image = toNumpyArr(bg_surf) / 255.0
            image = rotation(image, font_rotate)
            image = low_resolusion(image, sz)
            image = expand(image, sz)

            # 1ch
            image = rgb2grey(image).astype(np.float32)
            # 3ch
            #image = image.astype(np.float32)
            
            h, w = image.shape
            """ 描画されないものは除く """
            if len(np.where(image > 0.95)[0]) != h*w:
                # 1ch or 3ch
                data.append(image.reshape([h, w, 1]))
                #data.append(image)
                
                labels.append(id)
                # debug
                #cv2.imshow('sample', image)
                #cv2.waitKey(500)
            else:
                pass

        try:
            data = np.transpose(np.asarray(data).astype(np.float32), [0, 3, 1, 2])
            labels = np.asarray(labels).astype(np.int32)
            data_context = {'data': data,
                            'labels': labels}
            f.write('{} {}\n'.format(id, text.encode('utf-8')))
            with open(pklfile, 'w') as pkl:
                cPickle.dump(data_context, pkl, -1)
        except:
            print 'WARN id : {} , [ {} ] was ignored'.format(id, text.encode('utf-8'))

        
    f.close()
    progress.finish()
    
    
    """
    clock = pygame.time.Clock()
    
        while True:
            gameDisplay.fill(white)
            gameDisplay.blit(bg_surf, bg_rect)
        
            for event in pygame.event.get():
            
                if event.type == pygame.KEYDOWN:
                    #save_screen_img(bg_surf, 'hello.jpg')
                    print image.shape
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
            clock.tick(60)
    """
