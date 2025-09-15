
# assert decoder(llm(encoder("1 + 2"))) == 3
# assert decoder(llm(encoder("( 1 + 2 ) * 3"))) == 9


def train_llm_on_math(num_epochs=100, lr=0.001):
    """
    Train the LLM to perform mathematical computations.
    This demonstrates how an LLM learns to do math through examples.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Generate training data
    training_expressions = [
        generate(seed=i) for i in range(10)
    ]

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for expr in training_expressions:
            # Encode input and get target
            encoded_input = encoder(expr)
            target_value = vm(encoded_input)
            target_encoded = pack(NUMBER, target_value)
            target_token = target_encoded % VOCAB_SIZE

            # Prepare input and target tensors
            input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor([target_token], dtype=torch.long)

            # Forward pass
            logits = model(input_tensor)
            last_logits = logits[0, -1, :].unsqueeze(0)  # [1, vocab_size]

            # Compute loss
            loss = criterion(last_logits, target_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_expressions)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    print("Training completed!")


# Uncomment to train the model
# print("Training the LLM on mathematical expressions...")
# train_llm_on_math()



def train_reconstruction_only(num_epochs=100, lr=0.0001):
    """
    Phase 1 only: Train the model to reconstruct its inputs (autoencoder behavior)
    """
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=lr)
    training_expressions = [generate(seed=i) for i in range(20)]

    print("=== Training Input Reconstruction Only ===")
    new_model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for expr in training_expressions:
            encoded_input = encoder(expr)
            input_vectors = integers_to_vectors(encoded_input)
            input_tensor = input_vectors.unsqueeze(0)

            mu, sigma = new_model(input_tensor)

            # Reconstruct each position
            position_losses = []
            for pos in range(len(encoded_input)):
                target_bits = input_vectors[pos]
                pred_mu = mu[0, pos, :]
                reconstruction_loss = F.mse_loss(pred_mu, target_bits)
                position_losses.append(reconstruction_loss)

            loss = torch.stack(position_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_expressions)
            print(f"Epoch {epoch}, Reconstruction Loss: {avg_loss:.4f}")

    print("Reconstruction training completed!")


def train_completion_only(num_epochs=100, lr=0.0001):
    """
    Phase 2 only: Train the model to complete mathematical expressions
    (assumes model already understands input representation)
    """
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=lr)
    training_expressions = [generate(seed=i) for i in range(20)]

    print("=== Training Mathematical Completion Only ===")
    new_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for expr in training_expressions:
            encoded_input = encoder(expr)
            target_value = vm(encoded_input)
            target_encoded = pack(NUMBER, target_value)
            target_bits = torch.tensor(split_bits(target_encoded, VECTOR_SIZE), dtype=torch.float32)

            input_vectors = integers_to_vectors(encoded_input)
            input_tensor = input_vectors.unsqueeze(0)

            mu, sigma = new_model(input_tensor)
            last_mu = mu[0, -1, :]
            last_sigma = sigma[0, -1, :]

            # Check accuracy
            prediction = (last_mu > 0.5).long()

            def get_number(array):
                try:
                    tag, v = unpack(join_bits(array.long().tolist()))
                    return v
                except:
                    return -1

            if get_number(prediction) == get_number(target_bits):
                correct_predictions += 1
            total_predictions += 1

            # NLL loss for completion
            diff = target_bits - last_mu
            nll_loss = 0.5 * torch.sum(
                (diff ** 2) / (last_sigma ** 2) +
                torch.log(2 * math.pi * last_sigma ** 2)
            )

            loss = nll_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_expressions)
            accuracy = correct_predictions / total_predictions * 100
            print(f"Epoch {epoch}, Completion Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%")

        if correct_predictions / total_predictions > 0.9:
            print(f"Early stopping! Achieved {correct_predictions/total_predictions*100:.1f}% accuracy")
            break

    print("Completion training completed!")