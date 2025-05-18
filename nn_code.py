!pip install datasets
!pip install --upgrade datasets fsspec

from datasets import load_dataset
mnist_dataset = load_dataset("ylecun/mnist")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

class ClusteringAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(ClusteringAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, latent_size)
        )
        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, input_size),
            nn.Tanh()  # Output between -1 and 1 (normalized images)
        )

    def forward(self, x):

        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def get_embeddings(self, x):
        return self.encoder(x)
input_size = 784  # 28x28 images flattened
hidden_size = 128
latent_size = 64   # Size of bottleneck layer
learning_rate = 0.001
batch_size = 64
epochs = 5
n_clusters = 10    # Number of clusters for K-Means

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ClusteringAutoencoder(input_size, hidden_size, latent_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...")
train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        outputs, _ = model(data)
        loss = criterion(outputs, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()

def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), -1).to(device)
            _, latent = model(data)
            embeddings.append(latent.cpu().numpy())
            labels.append(target.numpy())

    return np.vstack(embeddings), np.concatenate(labels)

print("Extracting embeddings...")
train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

print("Applying K-Means clustering...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(train_embeddings)

test_clusters = kmeans.predict(test_embeddings)

# Evaluation Metrics
print("Evaluating clustering metrics...")
silhouette = silhouette_score(test_embeddings, test_clusters)
print(f"Silhouette Score: {silhouette:.4f}")

db_index = davies_bouldin_score(test_embeddings, test_clusters)
print(f"Davies-Bouldin Index: {db_index:.4f}")

ch_index = calinski_harabasz_score(test_embeddings, test_clusters)
print(f"Calinski-Harabasz Index: {ch_index:.4f}")

def compute_purity(clusters, labels):
    contingency_matrix = np.zeros((n_clusters, 10))  

    for i in range(len(clusters)):
        contingency_matrix[clusters[i], labels[i]] += 1

    cluster_sizes = contingency_matrix.sum(axis=1)
    max_label_counts = contingency_matrix.max(axis=1)
    purity = np.sum(max_label_counts) / np.sum(cluster_sizes)

    return purity, contingency_matrix

purity, contingency_matrix = compute_purity(test_clusters, test_labels)
print(f"Cluster Purity: {purity:.4f}")

# Visualize the clusters with t-SNE
print("Visualizing clusters using t-SNE...")
# Use a subset for t-SNE to save computation time
n_samples = 3000
subset_indices = np.random.choice(len(test_embeddings), n_samples, replace=False)
subset_embeddings = test_embeddings[subset_indices]
subset_clusters = test_clusters[subset_indices]
subset_labels = test_labels[subset_indices]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(subset_embeddings)

df_clusters = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'Cluster': subset_clusters
})

df_labels = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'Digit': subset_labels
})

# Plot clusters
plt.figure(figsize=(12, 10))
sns.scatterplot(x='x', y='y', hue='Cluster', data=df_clusters, palette='viridis', s=10)
plt.title(f't-SNE visualization of clusters\nSilhouette: {silhouette:.4f}, DB Index: {db_index:.4f}')
plt.savefig('tsne_clusters.png')
plt.close()

# Plot actual digit labels
plt.figure(figsize=(12, 10))
sns.scatterplot(x='x', y='y', hue='Digit', data=df_labels, palette='tab10', s=10)
plt.title('t-SNE visualization with actual digit labels')
plt.savefig('tsne_digits.png')
plt.close()

# Show example images from each cluster
plt.figure(figsize=(15, 10))
n_examples = 5

for cluster_id in range(n_clusters):
    cluster_indices = np.where(test_clusters == cluster_id)[0]
    if len(cluster_indices) >= n_examples:
        sample_indices = np.random.choice(cluster_indices, n_examples, replace=False)

        for i, idx in enumerate(sample_indices):
            plt.subplot(n_clusters, n_examples, cluster_id * n_examples + i + 1)
            img = test_dataset[idx][0].reshape(28, 28)
            plt.imshow(img.numpy(), cmap='gray')
            plt.title(f"C{cluster_id}: D{test_labels[idx]}")
            plt.axis('off')

plt.tight_layout()
plt.savefig('cluster_examples.png')
plt.close()

plt.figure(figsize=(10, 8))
contingency_df = pd.DataFrame(contingency_matrix,
                             index=[f'Cluster {i}' for i in range(n_clusters)],
                             columns=[f'Digit {i}' for i in range(10)])

row_sums = contingency_df.sum(axis=1)
normalized_contingency = contingency_df.div(row_sums, axis=0)

sns.heatmap(normalized_contingency, annot=True, cmap='viridis', fmt='.2f')
plt.title('Distribution of Digits in Each Cluster (Normalized)')
plt.tight_layout()
plt.savefig('cluster_distribution.png')
plt.close()

print("Analysis complete. Images saved.")
