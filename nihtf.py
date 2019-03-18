import numpy as np
import pandas as pd
import tensorflow as tf

def get_labels():
    label = pd.read_csv('Data_Entry_2017.csv',delimiter=",", usecols =[1])
    print('labels are processed')
    label = label.values
    label = label.T
    return label

def binary_labels(raw_labels):
    raw_labels = raw_labels.tolist()
    lyst=[]
    names = []
    for r in raw_labels:
        if "Pneumonia" in r:
            lyst.append(1)
            names.append('Pneumonia')
            continue
        else:
            lyst.append(0)
            names.append('Not Pneumonia')
        
    lyst = np.asarray(lyst)
    return lyst,names


def file_path():
    p = tf.gfile.Glob('/network/rit/lab/bhattacharya_lab/big45gbforgenerichomemadenetowrks/images/*.png')
    return p

def load_and_preprocess_alex(image):
    image = tf.read_file(image)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize_images(image,[227,227],align_corners=True)
    image = image/255.0
    return image

def alex_logits(in_data):
    input_layer = tf.reshape(in_data, -1)
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11], strides=4, padding="valid")
    lrn1 = tf.nn.lrn(input=conv1, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75)
    pool1_conv1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2)
    
    conv2 = tf.layers.conv2d(inputs=pool1_conv1, filters=256,kernel_size=[5, 5],strides=1,padding="same",activation=tf.nn.relu)
    lrn2 = tf.nn.lrn(input=conv2, depth_radius=5, bias=1.0, alpha=0.0001/5.0, beta=0.75)
    pool2_conv2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2)
    
    conv3 = tf.layers.conv2d(inputs=pool2_conv2, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    pool3_conv5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2, padding="valid")
    
    pool3_conv5_flat = tf.reshape(pool3_conv5, [-1, 6* 6 * 256])
    fc1 = tf.layers.dense(inputs=pool3_conv5_flat, units=4096, activation=tf.nn.relu)
    
    fc2 = tf.layers.dense(inputs=fc1, units=4096, activation=tf.nn.relu)
    
    logits = tf.layers.dense(inputs=fc2,units=2,name="logits_layer")
    
    return logits
    

path_point = file_path()
path_point = np.asarray(path_point)
print(len(path_point))
split = 0.8*len(path_point)
split = int(split)
#Point
train_path = path_point[:split]
vald_path = path_point[split:]

labels = get_labels()
labels_train=labels[:split]
labels_vald=labels[split:]

labels_train_b,labels_train_names = binary_labels(labels[:split])
labels_vald_b, labels_vald_names = binary_labels(labels[split:])
path_ds_train = tf.data.Dataset.from_tensor_slices(train_path)
path_ds_vald = tf.data.Dataset.from_tensor_slices(vald_path)

image_ds_train = path_ds_train.map(load_and_preprocess_alex, num_parallel_calls=10)
image_ds_vald = path_ds_vald.map(load_and_preprocess_alex, num_parallel_calls=10)

image_count_train = len(labels_train)
image_count_vald = len(labels_vald)
label_ds_train = tf.data.Dataset.from_tensor_slices(labels_train_b)
label_ds_vald = tf.data.Dataset.from_tensor_slices(labels_vald_b)
image_data_tagged_train = tf.data.Dataset.zip((image_ds_train,label_ds_train))
image_data_tagged_vald = tf.data.Dataset.zip((image_ds_vald,label_ds_vald))

BATCH_SIZE=10
img_batch_train = image_data_tagged_train.shuffle(buffer_size=image_count_train).repeat().batch(BATCH_SIZE)

img_batch_vald = image_data_tagged_vald.shuffle(buffer_size=image_count_vald).repeat().batch(BATCH_SIZE)

iterator_train = tf.data.Iterator.from_structure(img_batch_train.output_types,img_batch_train.output_shapes)
next_element_train = iterator_train.get_next()
training_init_op = iterator_train.make_initializer(img_batch_train)

iterator_vald = tf.data.Iterator.from_structure(img_batch_vald.output_types,img_batch_vald.output_shapes)
next_element_vald = iterator_vald.get_next()
validation_init_op = iterator_vald.make_initializer(img_batch_vald)

alx = alex_logits(next_element_train[0])

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=next_element_train[1], logits=alx))
optimizer = tf.train.AdamOptimizer().minimize(loss)

prediction = tf.argmax(alx,1)
equality = tf.equal(prediction, tf.arg_max(next_element_train[1],1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()

epochs = 15
avg_acc = 0
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    for i in range(epochs):
        L,_, acc = sess.run([loss,optimizer,accuracy])
        if i%50:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, L, acc * 100))
    #Validation
    valid_iters = 100
    sess.run(validation_init_op)
    for i in range(valid_iters):
        acc = sess.run([accuracy])
        avg_acc += acc
    print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,(avg_acc / valid_iters) * 100))
