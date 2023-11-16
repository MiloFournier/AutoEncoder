# AutoEncoder

## AutoEncoder for face image reconstruction

### Summary
| Section | Description |
| ------- | ----------- |
| 1. Project Description | Quick introduction |
| 2. Data Initialization | Blur + normalization |
| 3. Data Visualization | Print a row of images |
| 4. Model | AutoEncoder itself |
| 5. Results | Quick recap |

---
### 1. Project Description
This is a jupyter notebook where I implement an **Overcomplete Convolutional Denoising AutoEncoder in order to reconstruct face images**.

---
### 2. Data Initialization
To have better performances, it is a good idea to train the model on the images with noise. Here I added a Gaussian blur on every image:
```py
x_train_blur = np.empty_like(x_train)
for i in tqdm(range(len(x_train))):
    x_train_blur[i] = x_train[i]
    x_train_blur[i] = cv2.GaussianBlur(x_train_blur[i], (5, 5), 0)

x_valid_blur = np.empty_like(x_valid)
for i in tqdm(range(len(x_valid))):
    x_valid_blur[i] = x_valid[i]
    x_valid_blur[i] = cv2.GaussianBlur(x_valid_blur[i], (5, 5), 0)
```
Then I also normalize the data:
```py
mu, std = np.mean(x_train, axis=(0, 1, 2)), np.std(x_train, axis=(0, 1, 2))
print("mu: ", mu, " sigma: ", std)

def norm(a):
    return torch.from_numpy((a - mu) / std).permute(0, 3, 1, 2)

def denorm(a):
    return a.detach().cpu().permute(0, 2, 3, 1).numpy() * std + mu
```

---
### 3. Data Visualization 
Here I havea function to plot a row of images:
```py
def plot_rows(*img_rows, scale=1.):
    rows = len(img_rows)
    cols = len(img_rows[0])
    fig, axs = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
    for i, img_row in enumerate(img_rows):
        for j, im in enumerate(img_row):
            ax = axs[i, j]
            ax.imshow(np.clip(im, 0, 1))
            ax.axis('off')
    plt.tight_layout()
    plt.show()
```
We can visualize the input images and the ones blurred when running `plot_rows(x_train[:10], x_train_blur[:10])`:
<p align="center">
  <img width="900" alt="image" src="https://github.com/MiloFournier/AutoEncoder/assets/132404970/0c97d05e-ac9a-4abe-ab2b-6ebce4987d75">
</p>

---
### 4. Model
This model is an Overcomplete Convolutional Denoising AutoEncoder: 
```py
def build_model():
  layers = []
  # Encoder
  layers.append(nn.Conv2d(3, 16, 3))
  layers.append(nn.ReLU())
  layers.append(nn.Conv2d(16, 64, 3))
  layers.append(nn.ReLU())
  # Decoder
  layers.append(nn.ConvTranspose2d(64, 64, 3))
  layers.append(nn.ReLU())
  layers.append(nn.ConvTranspose2d(64, 16, 3))
  layers.append(nn.ReLU())
  layers.append(nn.Conv2d(16, 3, 1, 1))

  model = nn.Sequential(*layers)

  return model
```

---
### 5. Results
After training the model and running it for **15 epochs**, I have good results. I used the mean squared error loss function, and after 15 epochs it is **less than 2.75%**.

Here is the 1st epoch:
<p align="center">
  <img width="913" alt="image" src="https://github.com/MiloFournier/AutoEncoder/assets/132404970/05211429-134a-4894-aa42-9ad7b239434c">
</p>

And the 15th one:
<p align="center">
  <img width="913" alt="image" src="https://github.com/MiloFournier/AutoEncoder/assets/132404970/aae50ec5-b779-4b97-84a9-ef56b4e443ed">
</p>

<p align="center">
  <img width="596" alt="image" src="https://github.com/MiloFournier/AutoEncoder/assets/132404970/e053b8d9-86ac-414d-820b-02e0c1569cf3">
</p>

