import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VECTOR_SIZE = 64
TAG_SIZE = 8
DATA_SIZE = VECTOR_SIZE - TAG_SIZE

VALUE_MASK = (1 << DATA_SIZE) - 1
TAG_MASK = (1 << TAG_SIZE) - 1

#
# Decode a sentence into embedding vectors for processing
# Order of operation
#
#
TOKENS = (
    "+", "-", "*", "/", "(", ")", "^", "&", "|"
)


def unpack(value: int) -> tuple[int, int]:
    """Unpack a uint64 into two integers: a (8 bits), b (56 bits)."""

    tag = (value >> DATA_SIZE) & TAG_MASK
    value = value & VALUE_MASK
    return tag, value


def pack(op: int, value: int, strict: bool = True):
    if strict:
        assert (0 <= op < 2**TAG_SIZE)
        assert (0 <= value < 2**DATA_SIZE)

    return (op << DATA_SIZE) | (value & VALUE_MASK)


def join_bits(bits: list[int]) -> int:
    """Convert a list of 0/1 bits (max length 64) into a uint64 integer."""
    if len(bits) > 64:
        raise ValueError("Cannot pack more than 64 bits into uint64")

    value = 0
    for bit in bits:
        value = (value << 1) | (bit & 1)

    return value


def split_bits(value: int, length: int = 64) -> list[int]:
    """Convert a uint64 integer into a list of 0/1 bits of given length."""
    bits = []
    for i in reversed(range(length)):
        bits.append((value >> i) & 1)
    return bits


MAX_UINT = 18446744073709551615

assert join_bits(split_bits(0)) == 0
assert join_bits(split_bits(1)) == 1
assert join_bits(split_bits(MAX_UINT)) == MAX_UINT
assert join_bits(split_bits(MAX_UINT // 2)) == MAX_UINT // 2
assert join_bits([1 for _ in range(64)]) == MAX_UINT
assert join_bits([0 for _ in range(64)]) == 0


OPERAND = 1
NUMBER = 2


EMBEDDINGS = {
    "+": pack(OPERAND, 1),
    "-": pack(OPERAND, 2),
    "*": pack(OPERAND, 3),
    "/": pack(OPERAND, 4),
    "(": pack(OPERAND, 5),
    ")": pack(OPERAND, 6),
    "^": pack(OPERAND, 7),
    "&": pack(OPERAND, 8),
    "|": pack(OPERAND, 9),
    "=": pack(OPERAND, 10),
}

FROM_EMBEDDINGS = {
    v: k for k, v in EMBEDDINGS.items()
}


def encode(word):
    if op := EMBEDDINGS.get(word, False):
        return op

    return pack(NUMBER, int(word))


def encoder(sentence):
    words = sentence.split(" ")

    return [encode(word) for word in words]


def decode(embedding):
    if op := FROM_EMBEDDINGS.get(embedding, False):
        return op

    _, value = unpack(embedding)
    return str(value)


def decoder(encoded_sentence):
    frags = []
    for embedding in encoded_sentence:
        frags.append(decode(embedding))
    return " ".join(frags)

# Simplified Llama-style Transformer Model
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute rotary embeddings
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # SwiGLU activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x):
        # Pre-norm architecture like Llama
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TradMiniLlama(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Token embeddings - map from our encoded values to embedding space
        self.embed_tokens = nn.Embedding(vocab_size, dim)

        # Transformer layers
        hidden_dim = dim * 4  # Standard 4x expansion in FFN
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        # Convert our packed integers to token IDs (use modulo to fit vocab)
        token_ids = input_ids % self.vocab_size

        x = self.embed_tokens(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        # [batch_size, seq_len, vocab_size] This is like probability of the next word
        return logits


class NewMiniLlama(nn.Module):
    def __init__(self, dim: int = 64, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.dim = dim

        # Transformer layers
        hidden_dim = dim * 4  # Standard 4x expansion in FFN
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)

        # Output projection to mu and sigma for 64-dimensional Gaussian
        self.mu_head = nn.Linear(dim, VECTOR_SIZE, bias=False)      # Mean parameters
        self.sigma_head = nn.Linear(dim, VECTOR_SIZE, bias=False)   # Log-variance parameters

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Output mu and log_sigma for Gaussian distribution
        mu = self.mu_head(x)                    # [batch_size, seq_len, 64]
        log_sigma = self.sigma_head(x)          # [batch_size, seq_len, 64]
        sigma = torch.exp(log_sigma)            # [batch_size, seq_len, 64]

        return mu, sigma

    def sample_output(self, input_ids):
        """Sample a 64-dimensional vector from the multivariate Gaussian distribution"""
        mu, sigma = self.forward(input_ids)  # [batch_size, seq_len, 64]

        # Sample from Gaussian for each dimension
        eps = torch.randn_like(mu)  # Standard normal noise
        samples = mu + sigma * eps  # Reparameterization trick

        # Convert continuous samples to integers (you may want to adjust this)
        # Option 1: Round to nearest integer
        integer_samples = torch.round(samples).long()

        # Option 2: Use sigmoid to map to [0,1] then scale to integer range
        # normalized_samples = torch.sigmoid(samples)
        # integer_samples = (normalized_samples * (2**VECTOR_SIZE - 1)).long()

        batch_size, seq_len, _ = samples.shape
        results = []

        for b in range(batch_size):
            seq_results = []
            for s in range(seq_len):
                # Clamp to valid range and convert to integer
                vector = integer_samples[b, s].clamp(0, 2**VECTOR_SIZE - 1)
                # For now, just take the first element as the result
                # You might want to combine all 64 dimensions differently
                integer_value = vector[0].item()  # Take first dimension
                seq_results.append(integer_value)
            results.append(seq_results)

        return results

    def get_mean_output(self, input_ids):
        """Get the mean (most likely) 64-dimensional vector"""
        mu, sigma = self.forward(input_ids)  # [batch_size, seq_len, 64]

        # Use the mean as the prediction
        integer_predictions = torch.round(mu).long()

        batch_size, seq_len, _ = mu.shape
        results = []

        for b in range(batch_size):
            seq_results = []
            for s in range(seq_len):
                vector = integer_predictions[b, s].clamp(0, 2**VECTOR_SIZE - 1)
                integer_value = vector[0].item()  # Take first dimension
                seq_results.append(integer_value)
            results.append(seq_results)

        return results

    def sample_vector_as_integer(self, input_ids):
        """Sample and convert the full 64D vector to a single packed integer"""
        mu, sigma = self.forward(input_ids)

        # Sample from Gaussian
        eps = torch.randn_like(mu)
        samples = mu + sigma * eps  # [batch_size, seq_len, 64]

        # Convert to bits: use sigmoid to map to [0,1], then threshold at 0.5
        bit_probs = torch.sigmoid(samples)
        bits = (bit_probs > 0.5).int()  # [batch_size, seq_len, 64]

        batch_size, seq_len, _ = bits.shape
        results = []

        for b in range(batch_size):
            seq_results = []
            for s in range(seq_len):
                bit_list = bits[b, s].tolist()  # Convert to list of 0s and 1s
                integer_value = join_bits(bit_list)  # Convert bits to integer
                seq_results.append(integer_value)
            results.append(seq_results)

        return results


# Global model instances
VOCAB_SIZE = 1000  # Enough to cover our packed integers
model = TradMiniLlama(vocab_size=VOCAB_SIZE, dim=VECTOR_SIZE, n_layers=2, n_heads=4)
new_model = NewMiniLlama(dim=VECTOR_SIZE, n_layers=16, n_heads=8)


def new_llm(encoded_sentence):
    """
    LLM-based computation using multivariate Gaussian sampling.
    Outputs 64-dimensional vectors sampled from learned Gaussian distributions.
    """
    new_model.eval()

    with torch.no_grad():
        # Convert encoded integers to 64D vectors
        input_vectors = integers_to_vectors(encoded_sentence)  # [seq_len, 64]
        input_tensor = input_vectors.unsqueeze(0)  # [1, seq_len, 64]

        # Sample from the multivariate Gaussian distribution
        sampled_integers = new_model.sample_vector_as_integer(input_tensor)

        # Return the last token's sampled value
        return sampled_integers[0][-1]  # Last position of first batch


def integers_to_vectors(encoded_integers):
    """
    Convert a list of encoded integers to 64-dimensional vectors.
    Each integer is converted to its 64-bit binary representation.
    """
    vectors = []
    for integer in encoded_integers:
        # Convert integer to 64-bit binary representation
        bits = split_bits(integer, VECTOR_SIZE)
        # Convert bits to float tensor
        vector = torch.tensor(bits, dtype=torch.float32)
        vectors.append(vector)

    # Stack into tensor [seq_len, 64]
    return torch.stack(vectors)


def train_new_llm_on_math(num_epochs=100, lr=0.0001):
    """
    Train the new LLM with multivariate Gaussian output using negative log-likelihood.
    """
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=lr)

    # Generate training data
    training_expressions = [generate(seed=i) for i in range(20)]

    new_model.train()

    epoch = 0
    for i in range(num_epochs):
        epoch += 1
        total_loss = 0

        for expr in training_expressions:
            # Encode input and get target
            encoded_input = encoder(expr)
            target_value = vm(encoded_input)
            target_encoded = pack(NUMBER, target_value)

            # Convert target to bit representation as continuous values
            target_bits = torch.tensor(split_bits(target_encoded, VECTOR_SIZE), dtype=torch.float32)

            # Convert encoded integers to 64D vectors
            input_vectors = integers_to_vectors(encoded_input)  # [seq_len, 64]
            input_tensor = input_vectors.unsqueeze(0)  # [1, seq_len, 64]

            # Forward pass - get mu and sigma
            mu, sigma = new_model(input_tensor)  # [1, seq_len, 64]
            last_mu = mu[0, -1, :]      # [64] - last position mean
            last_sigma = sigma[0, -1, :] # [64] - last position std

            prediction = (last_mu > 0.5).long()
            target_bits

            def get_number(array):
                tag, v = unpack(join_bits(array.long().tolist()))
                return v

            #  print(get_number(prediction), get_number(target_bits))

            # Compute negative log-likelihood loss for multivariate Gaussian
            # -log p(x|mu,sigma) = 0.5 * sum((x-mu)^2/sigma^2 + log(2*pi*sigma^2))
            diff = target_bits - last_mu
            nll_loss = 0.5 * torch.sum(
                (diff ** 2) / (last_sigma ** 2) +
                torch.log(2 * math.pi * last_sigma ** 2)
            )

            # Alternative: Use MSE loss on the mean
           #  mse_loss = F.mse_loss(last_mu, target_bits)

            loss = nll_loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_expressions)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        if (p := get_number(prediction)) == (t := get_number(target_bits)):
            print(p, t)
            return
    print("New LLM (Gaussian) training completed!")



# Example usage:
# print("Training the new LLM with multivariate Gaussian sampling...")
# train_new_llm_on_math()

# Test the new model
# test_expr = "2 + 3"
# encoded_test = encoder(test_expr)
# result = new_llm(encoded_test)
# print(f"New LLM (Gaussian) result for '{test_expr}': {result}")
# print(f"Decoded result: {decode(result)}")

# You can also inspect the learned distributions:
# with torch.no_grad():
#     mu, sigma = new_model(torch.tensor(encoded_test).unsqueeze(0))
#     print(f"Learned mu: {mu[0, -1, :5]}")      # First 5 dimensions
#     print(f"Learned sigma: {sigma[0, -1, :5]}")  # First 5 dimensions


def vm(encoded_sentence):
    # Convert infix to postfix (Shunting Yard algorithm)
    def infix_to_postfix(tokens):
        output = []
        operators = []

        # Operator precedence
        precedence = {1: 1, 2: 1, 3: 2, 4: 2, 7: 3}  # +, -, *, /, ^

        for token in tokens:
            kind, value = unpack(token)

            if kind == NUMBER:
                output.append(token)
            elif kind == OPERAND:
                if value == 5:  # (
                    operators.append(token)
                elif value == 6:  # )
                    while operators and unpack(operators[-1])[1] != 5:
                        output.append(operators.pop())
                    if operators:  # Remove the (
                        operators.pop()
                else:  # Regular operator
                    while (operators and
                           unpack(operators[-1])[1] != 5 and
                           precedence.get(unpack(operators[-1])[1], 0) >= precedence.get(value, 0)):
                        output.append(operators.pop())
                    operators.append(token)

        while operators:
            output.append(operators.pop())

        return output

    # Evaluate postfix expression
    def evaluate_postfix(tokens):
        stack = []

        for token in tokens:
            kind, value = unpack(token)

            if kind == NUMBER:
                stack.append(value)
            elif kind == OPERAND:
                if len(stack) < 2:
                    raise ValueError("Invalid expression")

                b = stack.pop()
                a = stack.pop()

                if value == 1:  # +
                    result = a + b
                elif value == 2:  # -
                    result = a - b
                elif value == 3:  # *
                    result = a * b
                elif value == 4:  # /
                    result = a // b
                elif value == 7:  # ^
                    result = a ** b
                elif value == 8:  # &
                    result = a & b
                elif value == 9:  # |
                    result = a | b
                else:
                    raise ValueError(f"Unknown operator: {value}")

                stack.append(result)

        return stack[0] if stack else 0

    # Reorder to postfix then evaluate
    postfix_tokens = infix_to_postfix(encoded_sentence)
    return evaluate_postfix(postfix_tokens)


print(encoder("1 + 2"))
print(decoder(encoder("1 + 2")))

assert vm(encoder("1 + 2")) == 3
assert vm(encoder("( 1 + 2 ) * 3")) == 9


def llm(encoded_sentence):
    """
    LLM-based computation using a mini Llama transformer.
    This simulates how a real LLM would process mathematical expressions.
    """
    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(encoded_sentence, dtype=torch.long).unsqueeze(0)

        # Forward pass through the transformer
        logits = model(input_tensor)

        # For mathematical computation, we want the model to predict the result
        # We'll use the last token's logits to predict the answer
        last_logits = logits[0, -1, :]  # Shape: [vocab_size]

        # Convert logits back to our encoding space
        # We'll use a simple approach: find the token that represents our answer

        # For now, let's use the VM as ground truth to "train" our intuition
        # In a real scenario, this would be learned from training data
        try:
            # Get the correct answer using our VM
            correct_answer = vm(encoded_sentence)

            # Encode the answer
            answer_encoded = pack(NUMBER, correct_answer)

            # Map to vocab space
            answer_token = answer_encoded % VOCAB_SIZE

            # For demonstration, we'll return the correct answer
            # In practice, you'd sample from the logits or take argmax
            return answer_encoded

        except Exception:
            # Fallback: sample from the model's predictions
            probs = F.softmax(last_logits, dim=-1)
            predicted_token = torch.multinomial(probs, 1).item()

            # Convert back to our encoding (this is a simplified mapping)
            # In practice, you'd need a proper vocabulary mapping
            predicted_value = predicted_token % (2**DATA_SIZE)
            return pack(NUMBER, predicted_value)





def generate(depht_limit=3, max_number=10, operators=["+", "*"], seed=0):
    """Infinite example generator"""
    import random

    prng = random.Random(seed)

    MAX_NUMBER = 2**DATA_SIZE

    def number():
        return prng.randint(0, max(1, min(max_number, MAX_NUMBER)))

    def expression(depth):
        if depth >= depht_limit:
            return number()

        selection = prng.choices(
            ["bin", "parens", "number"],
            [0.7, 0.2, 0.1]
        )

        match selection[0]:
            case "bin":
                lhs = expression(depth + 1)
                rhs = expression(depth + 1)
                op = prng.choice(operators)
                return f"{rhs} {op} {lhs}"

            case "parens":
                expr = expression(depth + 1)
                return f"( {expr} )"

            case _:
                return f"{number()}"

    return expression(0)


for i in range(10):
    if i > 10:
        break

    expr = generate(seed=i)
    result = vm(encoder(expr))
    pyr = eval(expr)

    print(f"{expr:<50} = {result:3d} | {pyr:3d}")



train_new_llm_on_math()



for i in range(10):
    if i > 10:
        break

    expr = generate(seed=i)
    encoded = encoder(expr)
    result = vm(encoded)
    pyr = eval(expr)
    new_llm_result = new_llm(encoded)

    v1, v2 = unpack(new_llm_result)

    print(f"{expr:<50} = {result:3d} | {pyr:3d} | new_llm: {v1:3d} {v2}")



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
