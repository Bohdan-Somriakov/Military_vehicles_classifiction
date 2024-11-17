model = Sequential([
    Input(shape=(256, 256, 1)),
    Conv2D(32, (8,8), activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    
    Conv2D(64, (6,6), activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(128, (4,4), activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(128, (3,3), activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Flatten(),
    
    Dense(1024, activation = 'relu'),
    Dropout(0.3),
    
    Dense(512, activation = 'relu'),
    Dropout(0.3),
    
    Dense(256, activation = 'relu'),
    Dropout(0.3),
    
    Dense(train_generator.num_classes, activation = 'softmax')
])