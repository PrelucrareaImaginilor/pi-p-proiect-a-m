import cv2
import numpy as np
import os
import tensorflow as tf
from keras.src.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import target


def loadImage(image_path,target_size=(256,256)):
    images=[]
    for img in os.listdir(image_path):
        if img.endswith('.jpg'):
            img_path=os.path.join(image_path,img)
            img=load_img(img_path,color_mode="rgb")
            img=img.resize(target_size)
            img=img_to_array(img)/255.0#normalizeaza imaginea
            images.append(img)
    return np.array(images)

def loadMask(mask_path,target_size=(256,256)):
    masks=[]
    for mask in os.listdir(mask_path):
        if mask.endswith('.jpg'):
            m_path=os.path.join(mask_path,mask)
            mask=load_img(m_path,color_mode="grayscale")
            mask=mask.resize(target_size)
            mask=img_to_array(mask)/255.0#normalizeaza imaginea
            masks.append(mask)
    return np.array(masks)


def encoder_block(inputs, num_filters):
    # Convolution with 3x3 filter followed by ReLU activation
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)

    # Convolution with 3x3 filter followed by ReLU activation
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Max Pooling with 2x2 filter
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    # Upsampling with 2x2 filter
    x = tf.keras.layers.Conv2DTranspose(num_filters,(2, 2),strides=2,padding='same')(inputs)

    # Copy and crop the skip features
    # to match the shape of the upsampled input
    skip_features = tf.image.resize(skip_features,size=(x.shape[1],x.shape[2]))
    x = tf.keras.layers.Concatenate()([x, skip_features])

    # Convolution with 3x3 filter followed by ReLU activation
    x = tf.keras.layers.Conv2D(num_filters,3,padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Convolution with 3x3 filter followed by ReLU activation
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def unet_model(input_shape=(256, 256, 3), num_classes=1):
    inputs = tf.keras.layers.Input(input_shape)

    # Contracting Path
    s1 = encoder_block(inputs, 64)
    s2 = encoder_block(s1, 128)
    s3 = encoder_block(s2, 256)
    s4 = encoder_block(s3, 512)

    # Bottleneck
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same')(s4)
    b1 = tf.keras.layers.Activation('relu')(b1)
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same')(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)

    # Expansive Path
    s5 = decoder_block(b1, s4, 512)
    s6 = decoder_block(s5, s3, 256)
    s7 = decoder_block(s6, s2, 128)
    s8 = decoder_block(s7, s1, 64)

    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1,padding='same',activation='sigmoid')(s8)
    model = tf.keras.models.Model(inputs=inputs,outputs=outputs,name='U-Net')
    return model

if __name__=='__main__':
    image_dir=r"D:\Work\AC\PI\Proiect_PI\SetdeDate1\RMN"
    mask_dir=r"D:\Work\AC\PI\Proiect_PI\SetdeDate1\Mask"
    images=loadImage(image_dir)
    masks=loadMask(mask_dir)
    #plt.figure(figsize=(12, 12))
    model=unet_model(input_shape=(256,256,3),num_classes=1)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #compileaza modelul

    #antreneaza modelul
    #history=model.fit(images,masks,batch_size=8,epochs=50,validation_split=0.2)
    #print(images.shape)
    #print(masks.shape)
    #salveaza
    #model.save('unet_brain_tumor_segmentation')
    model = tf.keras.models.load_model("unet_brain_tumor_segmentation")
    imageTest_path=r"D:\Work\AC\PI\pythonProject\SetDateTest\RMN-TEST"
    maskTest_path=r"D:\Work\AC\PI\pythonProject\SetDateTest\MASK-TEST"
    testImg=loadImage(imageTest_path)
    testMask=loadMask(maskTest_path)
    #model.load_weights(r"D:\\Work\\AC\\PI\\pythonProject\\unet_brain_tumor_segmentation\\saved_model.pb")
    history=model.evaluate(testImg,testMask,8)
    print(f"Loss on test set: {history[0]}\n")
    print(f"Accuracy on test set: {history[1]}\n")
    #plt.plot(history.history['loss'],label='Train loss')
    #plt.plot(history.history['val_loss'],label='Validition loss')
    #plt.legend()
    #plt.show()
    prediction=model.predict(testImg,8)
    for i in range(50):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(testImg[i])
        plt.title('Imagine originala')
        plt.axis('off')

        #masca
        plt.subplot(1,3,2)
        plt.imshow(testMask[i],cmap='gray')
        plt.title('Masca reala')
        plt.axis('off')

        #predictia modelului
        plt.subplot(1,3,3)
        plt.imshow(prediction[i],cmap='gray')
        plt.title('Masca prezisa')
        plt.axis('off')

        plt.show()
