"""
Question Answering with DistilBERT
A complete implementation of a question answering system using fine-tuned DistilBERT
"""

import os

# Limit native math library threading to avoid startup memory failures on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import f1_score
from transformers import DistilBertTokenizerFast

PT_AVAILABLE = True
PT_IMPORT_ERROR = ""
try:
    import torch
except Exception as exc:
    torch = None
    PT_AVAILABLE = False
    PT_IMPORT_ERROR = str(exc)

TF_AVAILABLE = True
TF_IMPORT_ERROR = ""
try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(exc)

# For Jupyter notebooks (comment out if running as script)
# %matplotlib inline


# ============================================
# DATA LOADING AND EXPLORATION
# ============================================

def load_babi_dataset(data_path='data/'):
    """
    Load the bAbI question answering dataset
    
    Arguments:
        data_path -- path to the dataset folder
    
    Returns:
        babi_dataset -- loaded dataset object
    """
    babi_dataset = load_from_disk(data_path)
    return babi_dataset


def print_first_example(dataset):
    """
    Print the first example in the training set
    
    Arguments:
        dataset -- loaded dataset object
    """
    print("First training example:")
    print(dataset['train'][0])


def print_example_by_index(dataset, index=102):
    """
    Print a specific example by index
    
    Arguments:
        dataset -- loaded dataset object
        index -- index of the example to print
    """
    print(f"\nExample at index {index}:")
    print(dataset['train'][index])


def get_story_types(dataset):
    """
    Get unique story types in the dataset
    
    Arguments:
        dataset -- loaded dataset object
    
    Returns:
        type_set -- set of unique story types
    """
    type_set = set()
    for story in dataset['train']:
        if str(story['story']['type']) not in type_set:
            type_set.add(str(story['story']['type']))
    
    print(f"\nUnique story types: {type_set}")
    return type_set


# ============================================
# DATA PREPROCESSING
# ============================================

def flatten_dataset(dataset):
    """
    Flatten the nested structure of the dataset
    
    Arguments:
        dataset -- loaded dataset object
    
    Returns:
        flattened_babi -- flattened dataset
    """
    flattened_babi = dataset.flatten()
    return flattened_babi


def get_question_and_facts(story):
    """
    Extract question, context sentences, and answer from a story
    
    Arguments:
        story -- a story from the dataset
    
    Returns:
        dict containing question, sentences, and answer
    """
    dic = {}
    dic['question'] = story['story.text'][2]
    dic['sentences'] = ' '.join([story['story.text'][0], story['story.text'][1]])
    dic['answer'] = story['story.answer'][2]
    return dic


def get_start_end_idx(story):
    """
    Find the start and end indices of the answer within the context
    
    Arguments:
        story -- processed story with sentences and answer
    
    Returns:
        dict containing str_idx and end_idx
    """
    str_idx = story['sentences'].find(story['answer'])
    end_idx = str_idx + len(story['answer'])
    return {'str_idx': str_idx, 'end_idx': end_idx}


# ============================================
# TOKENIZATION AND ALIGNMENT
# ============================================

def initialize_tokenizer(tokenizer_path='tokenizer/'):
    """
    Initialize the DistilBERT tokenizer
    
    Arguments:
        tokenizer_path -- path to the tokenizer directory
    
    Returns:
        tokenizer -- DistilBertTokenizerFast instance
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


def tokenize_align(example, tokenizer, model_max_length=512):
    """
    Tokenize and align answer positions for the model
    
    Arguments:
        example -- processed example with sentences, question, and answer indices
        tokenizer -- DistilBERT tokenizer
        model_max_length -- maximum sequence length
    
    Returns:
        dict containing input_ids, attention_mask, start_positions, end_positions
    """
    encoding = tokenizer(
        example['sentences'],
        example['question'],
        truncation=True,
        padding=True,
        max_length=model_max_length
    )
    
    start_positions = encoding.char_to_token(example['str_idx'])
    end_positions = encoding.char_to_token(example['end_idx'] - 1)
    
    if start_positions is None:
        start_positions = model_max_length
    if end_positions is None:
        end_positions = model_max_length
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'start_positions': start_positions,
        'end_positions': end_positions
    }


# ============================================
# TENSORFLOW MODEL TRAINING
# ============================================

def create_tf_datasets(qa_dataset, tokenizer, batch_size=8):
    """
    Create TensorFlow datasets for training
    
    Arguments:
        qa_dataset -- preprocessed QA dataset
        tokenizer -- DistilBERT tokenizer (for reference)
        batch_size -- batch size for training
    
    Returns:
        train_tfdataset -- TensorFlow dataset for training
    """
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    train_ds = qa_dataset['train']
    
    columns_to_return = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    train_ds.set_format(type='tf', columns=columns_to_return)
    
    train_features = {
        "input_ids": tf.convert_to_tensor(np.array(train_ds["input_ids"]), dtype=tf.int32),
        "attention_mask": tf.convert_to_tensor(np.array(train_ds["attention_mask"]), dtype=tf.int32),
    }
    # Build dense int32 label tensors explicitly to avoid scalar-tensor shape issues.
    start_positions = tf.convert_to_tensor(np.array(train_ds['start_positions']), dtype=tf.int32)
    end_positions = tf.convert_to_tensor(np.array(train_ds['end_positions']), dtype=tf.int32)
    train_labels = {
        "start_positions": tf.expand_dims(start_positions, axis=-1),
        "end_positions": tf.expand_dims(end_positions, axis=-1),
    }
    
    train_tfdataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(batch_size)
    
    return train_tfdataset


def train_tensorflow_model(model, train_tfdataset, epochs=3, learning_rate=3e-5):
    """
    Train the TensorFlow DistilBERT model
    
    Arguments:
        model -- TFDistilBertForQuestionAnswering model
        train_tfdataset -- TensorFlow training dataset
        epochs -- number of training epochs
        learning_rate -- learning rate for Adam optimizer
    
    Returns:
        losses -- list of training losses
        model -- trained model
    """
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    loss_fn1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        print(f"Starting epoch: {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_tfdataset):
            with tf.GradientTape() as tape:
                answer_start_scores, answer_end_scores = model(x_batch_train)
                loss_start = loss_fn1(y_batch_train['start_positions'], answer_start_scores)
                loss_end = loss_fn2(y_batch_train['end_positions'], answer_end_scores)
                loss = 0.5 * (loss_start + loss_end)
            
            losses.append(loss)
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            
            if step % 20 == 0:
                print(f"Training loss (for one batch) at step {step}: {float(loss_start):.4f}")
    
    return losses, model


def plot_training_loss(losses):
    """
    Plot the training loss curve
    
    Arguments:
        losses -- list of training losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss over Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================
# INFERENCE FUNCTIONS
# ============================================

def predict_answer_tf(model, tokenizer, question, text):
    """
    Predict answer using TensorFlow model
    
    Arguments:
        model -- trained TFDistilBertForQuestionAnswering model
        tokenizer -- DistilBERT tokenizer
        question -- question string
        text -- context text string
    
    Returns:
        answer -- predicted answer string
    """
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    input_dict = tokenizer(text, question, return_tensors='tf')
    outputs = model(input_dict)
    start_logits = outputs[0]
    end_logits = outputs[1]
    
    all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    answer = ' '.join(all_tokens[
        tf.math.argmax(start_logits, 1)[0]:tf.math.argmax(end_logits, 1)[0] + 1
    ])
    
    return answer.capitalize()


def predict_answer_pt(model, tokenizer, question, text, device):
    """
    Predict answer using PyTorch model
    
    Arguments:
        model -- PyTorch DistilBertForQuestionAnswering model
        tokenizer -- DistilBERT tokenizer
        question -- question string
        text -- context text string
        device -- torch device (cuda/cpu)
    
    Returns:
        answer -- predicted answer string
    """
    if not PT_AVAILABLE:
        raise RuntimeError(f"PyTorch unavailable: {PT_IMPORT_ERROR}")

    input_dict = tokenizer(text, question, return_tensors='pt')
    input_ids = input_dict['input_ids'].to(device)
    attention_mask = input_dict['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    start_logits = outputs[0]
    end_logits = outputs[1]
    
    all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    answer = ' '.join(all_tokens[
        torch.argmax(start_logits, 1)[0]:torch.argmax(end_logits, 1)[0] + 1
    ])
    
    return answer.capitalize()


def run_interactive_qa(model, tokenizer, processed_dataset, framework='tf', device=None):
    """
    Run an interactive question-answering loop for the user.

    Arguments:
        model -- trained QA model (TF or PT)
        tokenizer -- DistilBERT tokenizer
        processed_dataset -- processed dataset containing question/sentences fields
        framework -- 'tf' or 'pt' to select prediction backend
        device -- torch device if framework is 'pt'
    """
    if model is None:
        print("\nNo model available for interactive QA.")
        return

    print("\n" + "=" * 60)
    print("Interactive Question Answering")
    print("=" * 60)
    print("Choose an option:")
    print("  1, 2, 3 : use a sample question/context")
    print("  c       : type your own context + question")
    print("  q       : quit interactive mode")

    sample_indices = [0, 187, 345]
    samples = []
    for idx in sample_indices:
        item = processed_dataset['test'][idx]
        samples.append(
            {
                "question": item["question"],
                "context": item["sentences"],
                "answer": item["answer"],
            }
        )

    while True:
        print("\nSample options:")
        for i, sample in enumerate(samples, start=1):
            print(f"  {i}) Q: {sample['question']}")
            print(f"     C: {sample['context']}")

        choice = input("\nEnter choice (1/2/3/c/q): ").strip().lower()
        if choice == "q":
            print("Exiting interactive mode.")
            break

        if choice in {"1", "2", "3"}:
            selected = samples[int(choice) - 1]
            question = selected["question"]
            context = selected["context"]
            expected = selected["answer"]
            print(f"\nQuestion: {question}")
            print(f"Context: {context}")
        elif choice == "c":
            context = input("Type context: ").strip()
            question = input("Type question: ").strip()
            expected = None
        else:
            print("Invalid option. Try again.")
            continue

        if not context or not question:
            print("Context and question are required. Try again.")
            continue

        try:
            if framework == 'pt':
                answer = predict_answer_pt(model, tokenizer, question, context, device)
            else:
                answer = predict_answer_tf(model, tokenizer, question, context)
            print(f"Predicted answer: {answer}")
            if expected is not None:
                print(f"Expected answer: {expected}")
        except Exception as inference_error:
            print(f"Inference error: {inference_error}")


# ============================================
# PYTORCH MODEL TRAINING
# ============================================

def setup_pytorch_datasets(qa_dataset):
    """
    Set up PyTorch datasets for training
    
    Arguments:
        qa_dataset -- preprocessed QA dataset
    
    Returns:
        train_ds -- PyTorch training dataset
        test_ds -- PyTorch test dataset
    """
    if not PT_AVAILABLE:
        raise RuntimeError(f"PyTorch unavailable: {PT_IMPORT_ERROR}")

    columns_to_return = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    
    train_ds = qa_dataset['train']
    test_ds = qa_dataset['test']
    
    train_ds.set_format(type='pt', columns=columns_to_return)
    test_ds.set_format(type='pt', columns=columns_to_return)
    
    return train_ds, test_ds


def compute_metrics(pred):
    """
    Compute F1 scores for start and end position predictions
    
    Arguments:
        pred -- prediction object from Trainer
    
    Returns:
        dict containing f1_start and f1_end scores
    """
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)
    
    f1_start = f1_score(start_labels, start_preds, average='macro')
    f1_end = f1_score(end_labels, end_preds, average='macro')
    
    return {
        'f1_start': f1_start,
        'f1_end': f1_end,
    }


def train_pytorch_model(pytorch_model, train_ds, test_ds, output_dir='results', epochs=3):
    """
    Train the PyTorch DistilBERT model using Hugging Face Trainer
    
    Arguments:
        pytorch_model -- DistilBertForQuestionAnswering model
        train_ds -- PyTorch training dataset
        test_ds -- PyTorch test dataset
        output_dir -- directory to save results
        epochs -- number of training epochs
    
    Returns:
        trainer -- trained Trainer object
    """
    if not PT_AVAILABLE:
        raise RuntimeError(f"PyTorch unavailable: {PT_IMPORT_ERROR}")

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir=None,
        logging_steps=50
    )
    
    trainer = Trainer(
        model=pytorch_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    return trainer


def get_device():
    """
    Get the available device (CUDA if available, else CPU)
    
    Returns:
        device -- torch device
    """
    if not PT_AVAILABLE:
        raise RuntimeError(f"PyTorch unavailable: {PT_IMPORT_ERROR}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    return device


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Question Answering with DistilBERT")
    print("=" * 60)
    
    # ========== LOAD DATASET ==========
    print("\nLoading bAbI dataset...")
    babi_dataset = load_babi_dataset('data/')
    
    # ========== EXPLORE DATASET ==========
    print("\nExploring dataset...")
    print_first_example(babi_dataset)
    print_example_by_index(babi_dataset, 102)
    
    # ========== GET STORY TYPES ==========
    type_set = get_story_types(babi_dataset)
    
    # ========== FLATTEN DATASET ==========
    print("\nFlattening dataset structure...")
    flattened_babi = flatten_dataset(babi_dataset)
    print(f"First example after flattening: {next(iter(flattened_babi['train']))}")
    
    # ========== EXTRACT QUESTION AND FACTS ==========
    print("\nExtracting questions and facts...")
    processed = flattened_babi.map(get_question_and_facts)
    print(f"Processed train example: {processed['train'][2]}")
    print(f"Processed test example: {processed['test'][2]}")
    
    # ========== FIND ANSWER INDICES ==========
    print("\nFinding answer indices in context...")
    processed = processed.map(get_start_end_idx)
    
    num = 187
    print(f"\nExample {num}:")
    print(processed['test'][num])
    start_idx = processed['test'][num]['str_idx']
    end_idx = processed['test'][num]['end_idx']
    print(f"Extracted answer: {processed['test'][num]['sentences'][start_idx:end_idx]}")
    
    # ========== INITIALIZE TOKENIZER ==========
    print("\nInitializing tokenizer...")
    tokenizer = initialize_tokenizer('tokenizer/')
    
    # ========== TOKENIZE AND ALIGN ==========
    print("\nTokenizing and aligning answers...")
    qa_dataset = processed.map(
        lambda x: tokenize_align(x, tokenizer, tokenizer.model_max_length)
    )
    
    # Remove unnecessary columns
    qa_dataset = qa_dataset.remove_columns([
        'story.answer', 'story.id', 'story.supporting_ids', 'story.text', 'story.type'
    ])
    
    print(f"Tokenized example: {qa_dataset['train'][200]}")
    
    # ========== SPLIT DATASETS ==========
    train_ds = qa_dataset['train']
    test_ds = qa_dataset['test']
    
    tf_model = None
    if TF_AVAILABLE:
        try:
            from transformers import TFDistilBertForQuestionAnswering

            # ========== TENSORFLOW MODEL TRAINING ==========
            print("\n" + "=" * 60)
            print("Training TensorFlow Model")
            print("=" * 60)

            print("\nLoading TensorFlow DistilBERT model...")
            tf_model = TFDistilBertForQuestionAnswering.from_pretrained("model/tensorflow", return_dict=False)

            print("\nCreating TensorFlow datasets...")
            train_tfdataset = create_tf_datasets(qa_dataset, tokenizer, batch_size=8)

            print("\nTraining TensorFlow model...")
            losses, tf_model = train_tensorflow_model(tf_model, train_tfdataset, epochs=3, learning_rate=3e-5)

            print("\nPlotting training loss...")
            plot_training_loss(losses)

            # ========== TENSORFLOW INFERENCE ==========
            print("\nTesting TensorFlow model inference...")
            question = 'What is south of the bedroom?'
            text = 'The hallway is south of the garden. The garden is south of the bedroom.'
            answer = predict_answer_tf(tf_model, tokenizer, question, text)
            print(f"Question: {question}")
            print(f"Answer: {answer}")

            print("\nTensorFlow model ready for interactive use.")
        except Exception as tf_runtime_error:
            print("\nSkipping TensorFlow section due to runtime error.")
            print(f"TensorFlow runtime error: {tf_runtime_error}")
    else:
        print("\nSkipping TensorFlow section because import failed.")
        print(f"TensorFlow error: {TF_IMPORT_ERROR}")
    
    pt_model = None
    pt_device = None
    if PT_AVAILABLE:
        from transformers import DistilBertForQuestionAnswering

        # ========== PYTORCH MODEL TRAINING ==========
        print("\n" + "=" * 60)
        print("Training PyTorch Model")
        print("=" * 60)

        print("\nSetting up PyTorch datasets...")
        train_ds_pt, test_ds_pt = setup_pytorch_datasets(qa_dataset)

        print("\nLoading PyTorch DistilBERT model...")
        pt_model = DistilBertForQuestionAnswering.from_pretrained("model/pytorch")

        print("\nTraining PyTorch model with Trainer...")
        trainer = train_pytorch_model(pt_model, train_ds_pt, test_ds_pt, output_dir='results', epochs=3)

        print("\nEvaluating on test set...")
        eval_results = trainer.evaluate(test_ds_pt)
        print(f"Evaluation results: {eval_results}")

        # ========== PYTORCH INFERENCE ==========
        print("\nTesting PyTorch model inference...")
        pt_device = get_device()
        pt_model.to(pt_device)

        question = 'What is east of the hallway?'
        text = 'The kitchen is east of the hallway. The garden is south of the bedroom.'
        answer = predict_answer_pt(pt_model, tokenizer, question, text, pt_device)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print("\nSkipping PyTorch section because import failed.")
        print(f"PyTorch error: {PT_IMPORT_ERROR}")

    # ========== INTERACTIVE QA ==========
    if pt_model is not None:
        run_interactive_qa(pt_model, tokenizer, processed, framework='pt', device=pt_device)
    elif tf_model is not None:
        run_interactive_qa(tf_model, tokenizer, processed, framework='tf')
    else:
        print("\nSkipping interactive QA because no model is available.")
    
    # ========== SAVE MODELS ==========
    print("\nSaving models...")
    save_tf = input("\nSave TensorFlow model? (y/n): ").strip().lower()
    if save_tf == 'y' and tf_model is not None:
        tf_model.save_pretrained('saved_models/tensorflow/')
        print("TensorFlow model saved to 'saved_models/tensorflow/'")
    elif save_tf == 'y':
        print("TensorFlow model is not available to save in this run.")
    
    save_pt = input("\nSave PyTorch model? (y/n): ").strip().lower()
    if save_pt == 'y' and pt_model is not None:
        pt_model.save_pretrained('saved_models/pytorch/')
        print("PyTorch model saved to 'saved_models/pytorch/'")
    elif save_pt == 'y':
        print("PyTorch model is not available to save in this run.")
    
    print("\nProgram completed successfully!")


if __name__ == "__main__":
    main()