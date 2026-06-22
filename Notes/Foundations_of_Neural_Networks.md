# 02 — Foundations of Neural Networks
> 3Blue1Brown "But what is a neural network?" — discussion notes  
> Week 1

---

## Why Neural Networks?

Kuch cheezein hain jo humans easily karte hain but rules mein likhna impossible hai.

Example: Handwritten "3" recognize karna. Har insaan alag likhta hai — mota, patla, tilted. Koi ek rule nahi ban sakta jo sab pakad le.

**Solution:** Model ko rules mat do. Examples do, aur seekhne do.

---

## Architecture — Number Recognition Example

Input: 28×28 pixel image = **784 pixels total**

```
Input Layer        Hidden Layer(s)      Output Layer
(784 neurons)  →   (e.g. 16 neurons) →  (10 neurons)
one per pixel      feature detectors    one per digit (0–9)
```

**Output mein jo neuron sabse zyada activate ho = woh digit answer hai.**

---

## Activation

Har neuron ki ek value hoti hai — **0 to 1** ke beech.

- 1 ke karib = neuron "on" hai, kuch detect kiya
- 0 ke karib = neuron "off" hai

Pixel ke liye — brightness ko 0–1 mein represent karo. Yahi activation hai.

---

## Weights — Importance of Each Connection

Har neuron ke aage wale layer se connections hain. Har connection pe ek **weight** hota hai.

- **Positive weight** → "yeh pixel on ho toh mujhe care karna chahiye"
- **Negative weight** → "yeh pixel on ho toh shayad yeh pattern nahi hai"
- **Weight near 0** → "yeh pixel mujhe matter nahi karta"

**Weights kisi ne set nahi kiye — model ne training mein khud seekhe.**

---

## Bias — Personal Threshold

Weight sum ke baad ek **bias** add hota hai.

- Bias controls karta hai ki neuron kitni aasaani se fire kare
- High bias → easily activate hoga
- Low/negative bias → zyada evidence chahiye activate hone ke liye

**Har neuron ka apna alag bias hota hai.** Bias koi bhi real number ho sakta hai — 0 to 1 tak limited nahi.

---

## Sigmoid Function

Formula:
```
output = sigmoid( w1*a1 + w2*a2 + ... + w784*a784 + bias )
```

Sigmoid ka kaam — koi bhi number lo, output **0 to 1** ke beech aa jaaye.

Isliye useful hai — activations 0–1 mein rakhne hain, but weighted sum koi bhi value le sakta hai.

---

## Hidden Layers — Sub-questions

Seedha input → output connect karo toh model crude hoga.

Hidden layer **beech mein chhote chhote features detect** karta hai.

Example — "3" detect karne ke liye:
- Neuron A: "Kya upar curve hai?"
- Neuron B: "Kya neeche curve hai?"
- Neuron C: "Kya beech mein gap hai?"

Output layer in sub-answers ko milata hai → final answer.

**Important:** Humne nahi bataya ki kaunsa neuron kya detect kare. Model ne khud training mein figure out kiya. Kabhi kabhi humans bhi explain nahi kar paate ki ek hidden neuron exactly kya pakad raha hai.

---

## Hidden Layer Size

Kitne neurons? — **Hyperparameter** hai, koi fixed formula nahi.

- Bohot kam → **underfitting** — complex patterns nahi pakad paata
- Bohot zyada → **overfitting** — training data ratt leta hai, naye data pe fail karta hai

3B1B ke example mein 16 liye — simplicity ke liye, not a rule.

---

## Training — Model Khud Kaise Seekhta Hai

1. **Random start** — sab weights aur biases random numbers se shuru
2. **Galti measure karo** — formula se pata karo kitna galat tha → **Loss**
3. **Adjust karo** — weights aur biases thoda change karo taaki loss kam ho → **Backpropagation**
4. **Repeat** — lakho examples pe

### Gradient Descent — Intuition

> Pahaad pe khadi ho, aankhein band, valley mein jaana hai.  
> Bas itna feel kar sakti ho ki zameen kis taraf dhaali hai.  
> Us taraf kadam lo. Repeat. Dheere dheere valley mili.

- Pahaad = Loss
- Valley = Minimum loss
- Dhalan = Gradient
- Kadam = Weight update

**Kisi ne rule nahi bataya. Bas ek goal diya — galti kam karo. Baaki math ne handle kiya.**

---

## Brain vs Neural Network

| | Human Brain | Neural Network |
|---|---|---|
| Neurons | ~86 billion | Hundreds to millions |
| Learning | Implicit, lifelong | Explicit, labeled data |
| "Weights" | Synapse strength | Numbers in a matrix |
| Needs | Thodi si examples | Laakho labeled examples |
| Understanding | Meaning + connection | Pattern matching |

Brain se **inspired** hai NN — exact copy nahi. Brain ka learning mechanism abhi bhi open research question hai.

---

## Key Insight

> Formula ragad ke mat seekho — meaning se seekho.  
> Weights = importance. Bias = threshold. Hidden layer = sub-questions. Training = galti se adjust karna.  
> Yeh sticky hai. Formula nahi.

---

## Next

- [ ] Transformer architecture — attention mechanism
- [ ] Embeddings — words ko numbers mein represent karna
- [ ] Backpropagation deeper (optional — 3B1B next video)