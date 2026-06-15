def align_labels_with_tokens(labels, word_ids):
    print('labels', labels)
    print('word_ids', word_ids)
    new_labels = []
    current_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens ([CLS], [SEP])
            new_labels.append(0)
        elif word_idx != current_word_idx:
            # This is the FIRST subword of a new word
            new_labels.append(labels[word_idx])
        else:
            new_labels.append(0)
            #new_labels.append(labels[word_idx])
            
        current_word_idx = word_idx
    
    return new_labels


def get_entities(tokenizer, input_ids, predictions, labels):
    words = tokenizer.convert_ids_to_tokens(input_ids)

    def retrieve_tokens(predictions):
        entities = []
        current_entity_tokens = []
        
        for word, pred in zip(words, predictions):
            is_subword = word.startswith("##") or word.startswith("#")
            clean_word = word.lstrip("#")

            if pred == 1:
                # If we were already building an entity, save it before starting a new one
                if current_entity_tokens:
                    entities.append(" ".join(current_entity_tokens))
                    current_entity_tokens = []
                current_entity_tokens.append(clean_word)
                
            elif pred == 2:
                if is_subword and current_entity_tokens:
                    current_entity_tokens[-1] += clean_word
                else:
                    current_entity_tokens.append(clean_word)
                    
            elif pred == 0:
                if is_subword and current_entity_tokens:
                    # Even if pred is 0, if it's a subword of a labeled word, we attach it
                    current_entity_tokens[-1] += clean_word
                else:
                    if current_entity_tokens:
                        entities.append(" ".join(current_entity_tokens))
                        current_entity_tokens = []
        
        return entities
    
    entities = retrieve_tokens(predictions)
    entities_gold = retrieve_tokens(labels)

    return entities, entities_gold 


def get_entities_batch(tokenizer, input_ids_batch, predictions_batch, labels_batch, inference):
    """
    Retrieves entities for a batch of inputs.
    
    Args:
        tokenizer: The tokenizer used for encoding.
        input_ids_batch: Tensor of shape [batch_size, seq_len]
        predictions_batch: Tensor of shape [batch_size, seq_len]
        labels_batch: Tensor of shape [batch_size, seq_len]
        
    Returns:
        batch_entities: List of list of predicted entity strings per sample.
        batch_entities_gold: List of list of actual entity strings per sample.
    """
    
    # Helper to convert tensor to list if necessary
    def to_list(tensor_or_list):
        if hasattr(tensor_or_list, "tolist"):
            return tensor_or_list.tolist()
        return tensor_or_list

    # Ensure inputs are lists for easy iteration
    input_ids_batch = to_list(input_ids_batch)
    predictions_batch = to_list(predictions_batch)
    if not inference:
        labels_batch = to_list(labels_batch)

    batch_entities = []
    if not inference:
        batch_entities_gold = []

    # Iterate over the batch
    for i in range(len(input_ids_batch)):
        input_ids = input_ids_batch[i]
        preds = predictions_batch[i]
        if not inference:
            labs = labels_batch[i]

        # Convert IDs to tokens for this specific sequence
        # Skip special tokens usually handled here, but raw conversion is fine 
        # based on your original snippet.
        words = tokenizer.convert_ids_to_tokens(input_ids)

        def retrieve_tokens(sequence_predictions):
            entities = []
            current_entity_tokens = []
            
            for word, pred in zip(words, sequence_predictions):
                # 1. Identify if it's a subword and clean it
                # Note: Use word.startswith("##") for BERT or "Ġ" for RoBERTa
                is_subword = word.startswith("##") 
                clean_word = word.replace("##", "")
                
                if word in tokenizer.all_special_tokens:
                    continue

                # 2. Logic: Should we start a new entity?
                # A new entity starts if we hit a B-tag (pred == 1)
                if pred == 1:
                    if current_entity_tokens:
                        entities.append(" ".join(current_entity_tokens))
                        current_entity_tokens = []
                    current_entity_tokens.append(clean_word)

                # 3. Logic: Should we continue an entity?
                elif pred == 2: # I-tag
                    if not current_entity_tokens:
                        # Edge case: I-tag without a B-tag, treat as a B-tag
                        current_entity_tokens.append(clean_word)
                    elif is_subword:
                        # Glue subword directly to the last piece
                        current_entity_tokens[-1] += clean_word
                    else:
                        # Append as a new word in the same entity phrase
                        current_entity_tokens.append(clean_word)

                # 4. Logic: Outside an entity or a subword of the current word
                else: # pred == 0 (O-tag)
                    if is_subword and current_entity_tokens:
                        # IMPORTANT: Even if the model predicts 'O' for "##nary", 
                        # if it's a subword of a word that started as an entity, glue it.
                        current_entity_tokens[-1] += clean_word
                    else:
                        # We hit a true 'Outside' word, close the current entity
                        if current_entity_tokens:
                            entities.append(" ".join(current_entity_tokens))
                            current_entity_tokens = []
            
            if current_entity_tokens:
                entities.append(" ".join(current_entity_tokens))
            
            return entities
        
        # Process current sample
        sample_entities = retrieve_tokens(preds)
        if not inference:
            sample_gold = retrieve_tokens(labs)
        
        batch_entities.append(sample_entities)
        if not inference:
            batch_entities_gold.append(sample_gold)

    if not inference:
        return batch_entities, batch_entities_gold
    else:
        return batch_entities