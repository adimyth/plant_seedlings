if __name__=="__main__":
    from DeepModel import DeepModel
    import pandas as pd

    GLOBAL_PATH = "./"
    arg_dict = {
        "GLOBAL_SEED": 42,
        "GLOBAL_PATH": "./",
        "TRAIN_DIR": GLOBAL_PATH+'data/train/',
        "VALIDATION_DIR": GLOBAL_PATH+'data/validation/',
        "TEST_DIR": GLOBAL_PATH+'data/test/',
        "MODEL_PATH": GLOBAL_PATH+'models/',
        "MODEL_CHECKPOINT_PATH": GLOBAL_PATH+'checkpoints/',
        "OUTPUT_BASE_NAME": 'best_model.h5',
        "NAME": 'log.csv',
        "LOG_PATH": GLOBAL_PATH+"logs/",
        "IMG_HEIGHT": 80,
        "IMG_WIDTH": 80,
        "num_classes": 5,
        "ngpus": 1
        }

    params = {
        "learning_rate": 0.01,
        "epochs": 1,
        "batch_size": 32,
        "loss": 'categorical_crossentropy',
        "metrics": ['accuracy']
    }

    new_model = DeepModel(arg_dict, params, validation=True)
    new_model.build_nn().augment_data().train().predict().save_as_csv()