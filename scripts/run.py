import os
import elmo_helper
print(r"""
 _______ _       ______           _______            _                   
(_______) |     |  ___ \         (_______)          (_)                  
 _____  | |     | | _ | | ___     _        ____ ____ _ ____   ____  ____ 
|  ___) | |     | || || |/ _ \   | |      / ___) _  | |  _ \ / _  )/ ___)
| |_____| |_____| || || | |_| |  | |_____| |  ( ( | | | | | ( (/ /| |    
|_______)_______)_||_||_|\___/    \______)_|   \_||_|_|_| |_|\____)_|

                    *** Initializing ***

""")


def input_digit(msg):
    v = ''
    while not v.isdigit():
        v = input(msg)
    v = int(v)
    return v


def input_path(msg):
    v = ""
    while not os.path.exists(v):
        v = input(msg)
    return v


print(
    r"""
                     
|\/|  _   _|  _   _ 
|  | (_) (_| (/_ _> 

Vocab Creation:         2    
Training Preprocess:    3
Train ELMo:             5       
Testing Preprocess:     7  
Test ELMo:              11     
Dump weights:           13     

"""

)


mode = input_digit("Enter the mode: ")

CHECKPOINT_DIR = 'checkpoint'

if mode % 2 == 0:
    input("Press enter to create the vocab file...\n>")
    train_files_location = input_path("Train files directory: ")
    vocab_file_location = 'vocab.txt'
    max_word_length, n_train_tokens = elmo_helper.vocab.create_vocab_file(
        source_directory=train_files_location, output_vocab_file=vocab_file_location, batch_size=1000)
    print("Vocab file creation completed...")

if mode % 3 == 0:
    input("Press enter to process the train files...\n>")
    if mode % 2 != 0:
        train_files_location = input_path("Train files directory: ")
    elmo_helper.preprocess(directory=train_files_location,
                           save_directory=train_files_location + "_f", notify_multiplier=10)
    print("Processing the train files completed...")


if mode % 5 == 0:
    input("Press enter to train the model...\n>")
    if mode % 2 != 0:
        max_word_length = input_digit('Maximum word length: ')
        n_train_tokens = input_digit('Number of input tokens: ')
        vocab_file_location = input_path('Vacab file name: ')
    if mode % 2 != 0 or mode % 3 != 0:
        processed_train_files = input_path("Processed train files directory: ")
    batch_size = input_digit('Batch size (default is 16): ')
    n_gpus = input_digit('Gpus used (default is 1): ')

    elmo_helper.train_model(max_word_length=max_word_length, n_train_tokens=n_train_tokens, batch_size=batch_size, n_gpus=n_gpus,
                            train_files=processed_train_files + '/*', vocab_file=vocab_file_location, checkpoint_dir=CHECKPOINT_DIR)
    print("Training the model completed...")


if mode % 7 == 0:
    input("Press enter to process the test files...\n>")
    test_files_location = input_path("Test files directory: ")
    processed_test_files = test_files_location + "_f"
    elmo_helper.preprocess(directory=test_files_location,
                           save_directory=processed_test_files, notify_multiplier=10)
    print("Processing the test files completed...")


if mode % 11 == 0:
    input("Press enter to test the model...\n>")
    if mode % 5 != 0:
        batch_size = input_digit('Batch size (default is 16): ')
    if mode % 7 != 0:
        processed_test_files = input_path("Processed test files directory: ")
    if mode % 2 != 0 or mode % 5 != 0:
        vocab_file_location = input_path('Vacab file name: ')

    elmo_helper.run_test(batch_size=batch_size, test_files=processed_test_files +
                         '/*', vocab_file=vocab_file_location, checkpoint_dir=CHECKPOINT_DIR)
    print("Training the model completed...")


if mode % 13 == 0:
    input("Press enter to dump weights...\n>")
    elmo_helper.save_weights(checkpoint_dir=CHECKPOINT_DIR,
                             weights_file='weights.hdf5')
    print("Weight dumping completed...")
