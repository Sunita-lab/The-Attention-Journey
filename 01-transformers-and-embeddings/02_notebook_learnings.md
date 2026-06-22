# 03 — Embeddings: Words ko Numbers mein
> Practical exploration — DistilBERT embeddings  
> Week 1 

---

## Embedding kya hai

Model strings nahi samajhta — numbers samajhta hai.

Toh har word ko ek **list of numbers** mein convert karte hain. Yeh list = **vector** = word ka mathematical representation.

```
"king"  → [0.2, 0.8, -0.1, 0.5, ...]  (768 numbers)
"queen" → [0.1, 0.9, -0.2, 0.6, ...]  (768 numbers)
"apple" → [0.9, -0.3, 0.7, -0.1, ...]  (768 numbers)
```

Similar meaning wale words → vectors karib karib (similar direction)
Alag meaning wale words → vectors door (different direction)

---

## 768 Dimensions kyun?

3D space mein 3 numbers — x, y, z. Humne manually decide kiya kya hai.

768D mein — **model ne khud decide kiya** training mein ki kaunsi dimension kya capture kare. Humne sirf itna bola — "768 numbers mein represent karo."

Kabhi kabhi researchers probe karke roughly samajhte hain — "yeh dimensions gender capture kar rahi hain, yeh royalty" — but exactly kya hai har dimension mein? Black box hai.

Zyada dimensions = zyada nuance = better meaning capture.

---

## Cosine Similarity

Do vectors ke beech ka **angle** nikalta hai — distance nahi.

```
Angle 0°  → same direction → similarity 1.0 → identical meaning
Angle 90° → perpendicular  → similarity 0.0 → no relation
```

Angle kyun, distance kyun nahi? Kyunki meaning **direction** mein hoti hai, vector ke size mein nahi.

---

## PyTorch kya hai

Neural networks ke liye ek library — NumPy jaisi but powerful:
- GPU pe fast calculations
- Automatically gradients nikaal sakta hai — isliye training hoti hai
- Neural network layers ready made hain

---

## Code — Step by Step

```python
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# Model load karo
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
```

**HuggingFace token kyun nahi?**
Token sirf private/gated models ke liye chahiye (Gemma, Llama). DistilBERT public hai.

---

```python
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding
```

**`tokenizer(text, return_tensors="pt")`**
Text ko numbers mein convert karta hai — ek dictionary return karta hai:
```python
{
  "input_ids": tensor([[101, 2079, 102]]),       # word IDs
  "attention_mask": tensor([[1, 1, 1]])           # kaunse tokens real hain
}
```

**`torch.no_grad()`**
Gradients track mat karo — training nahi kar rahe, sirf embedding nikal rahe hain. Memory aur speed save hoti hai.

**`**inputs`**
Dictionary ko unpack karta hai — Python ka general feature:
```python
model(**inputs)
# same as:
model(input_ids=..., attention_mask=...)
```

**`outputs.last_hidden_state[:, 0, :]`**
Last layer ki output leti hai. `0` matlab CLS token — jo poore sentence ka summary vector hota hai DistilBERT mein.

**Output shape: `torch.Size([1, 768])`**
- `1` → ek sentence process kiya
- `768` → 768 dimensions ka vector

---

## Experiments aur Results

### Single words
```python
king  = get_embedding("king")
queen = get_embedding("queen")
apple = get_embedding("apple")
```

| Pair | Similarity |
|------|-----------|
| King — Queen | 0.9897 |
| King — Apple | 0.9486 |
| Queen — Apple | 0.9539 |

King-Queen sabse karib ✅ — but gap chhota hai. Single words mein context nahi hota.

---

### Context ke saath sentences
```python
s1 = "the king ruled the kingdom"
s2 = "the queen ruled the kingdom"
s3 = "I ate a fresh apple today"
```

| Pair | Similarity |
|------|-----------|
| King sentence — Queen sentence | 0.9918 |
| King sentence — Apple sentence | 0.9249 |

Gap zyada clear — context dene se separation better hota hai.

---

### Mirra-relevant experiment
```python
s1 = "This vaccine causes serious side effects"
s2 = "This vaccine leads to harmful reactions"   # same claim, alag words
s3 = "The stock market crashed today"            # alag topic
```

| Pair | Similarity |
|------|-----------|
| Same claim (alag words) | 0.9889 |
| Alag topic | 0.8974 |

**Yeh Mirra ke liye real use case hai.**

---

## TF-IDF vs Embeddings — Mirra ke liye

| | TF-IDF | Embeddings |
|---|---|---|
| "causes" vs "leads to" | Alag words = alag | Same meaning = karib |
| Paraphrased claims | Miss kar sakta hai | Pakad leta hai |
| Context | Nahi | Haan |
| Speed | Fast | Slower |

Mirra mein embeddings upgrade hone se **paraphrased misinformation** bhi pakad payega.

---

## Key Takeaways

- Word = vector = 768 numbers = ek point in 768D space
- Similar meaning → vectors karib → cosine similarity high
- Context dene se embeddings better hoti hain
- 768 dimensions model ne khud decide kiye — hum nahi jaante exactly kya hai har dimension mein
- `torch.no_grad()` = inference mode, training nahi
- `**inputs` = dictionary unpack karna

---

## Next

- [ ] Transformer architecture — attention mechanism
- [ ] Fine-tuning vs training from scratch
- [ ] Week 2 — LoRA & QLoRA