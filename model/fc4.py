import tensorflow as tf
slim = tf.contrib.slim

def preprocess_image(image):
    image = tf.clip_by_value(image, 0., 65535.)
    image = image / 65535.
    image = image * 255
    image = image[...,::-1]
    return image

def backbone(images, reuse, backbone_type="alexnet"):
    """ Scope names are hard-coded for loading pretrained weights.
        This could be either alexnet or squeezenet. 
    """
    assert backbone_type in ["alexnet", "squeezenet"]
    
    if backbone_type == "alexnet":
        from model.alexnet import backbone as _backbone
    elif backbone_type == "squeezenet":
        from model.squeezenet import backbone as _backbone
    
    with tf.variable_scope('AlexNet', reuse=tf.AUTO_REUSE):
        features = _backbone(images)
        
    return features

def illuminant_estimator(features, is_training, reuse):
    """ Scope names are hard-coded for loading pretrained weights.
    """
    with tf.variable_scope('branch', reuse=reuse):
        fc1 = slim.conv2d(features, 64, [6, 6], scope='fc1')
        fc1 = slim.dropout(fc1, keep_prob=0.5, is_training=is_training)
        illum_maps = slim.conv2d(fc1, 3, [1, 1], scope='fc2', activation_fn=tf.nn.relu) # RGB. (b, h, w, 3)
        illums = tf.reduce_sum(illum_maps, axis=(1, 2)) # RGB. (b, 3)
    return illums
        
def model(images, is_training, reuse, backbone_type='alexnet'):
    """ Scope names are hard-coded for loading pretrained weights.
        images is tensorflow placeholder (b, h, w, 3).
    """
    
    output = {}
    
    images = preprocess_image(images)
    
    with tf.variable_scope("FCN", reuse=reuse):
        output['features'] = backbone(images, reuse=reuse, backbone_type=backbone_type)
        output['illums'] = illuminant_estimator(output['features'], is_training=is_training, reuse=reuse)

    return output