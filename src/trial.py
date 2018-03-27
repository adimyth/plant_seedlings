if __name__=="__main__":
    from DeepModel import DeepModel
    import pandas as pd

    GLOBAL_PATH = "./"
    arg_dict = {
        "GLOBAL_SEED": 42,
        "GLOBAL_PATH": "./",
        "SAMPLE_PATH": GLOBAL_PATH+'data/sample_submission.csv',
        "TRAIN_PATH": GLOBAL_PATH+'data/train/',
        "VALIDATION_PATH": GLOBAL_PATH+'data/validation/',
        "TEST_PATH": GLOBAL_PATH+'data/test/',
        "MODEL_PATH": GLOBAL_PATH+'models/',
        "MODEL_CHECKPOINT_PATH": GLOBAL_PATH+'checkpoints/',
        "LOG_PATH": GLOBAL_PATH+"logs/",
        "IMG_HEIGHT": 80,
        "IMG_WIDTH": 80,
        "ngpus": 1
        }

    params = {
        "learning_rate": 0.001,
        "epochs": 30,
        "batch_size": 32,
        "loss": 'categorical_crossentropy',
        "metrics": ['accuracy']
    }

    new_model = DeepModel(arg_dict, params)#, create_validation=True)
    print("[INFO] count_label_freq: ", new_model.label_freq)
    image_count = sum(new_model.label_freq.values())
    print("[INFO] Total image count: ", image_count)
    
    new_model.nn_multigpu().setup_data().train().predict().make_submission_kaggle_seedling()
    new_model.save_run()