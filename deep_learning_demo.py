import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_layer_net import DeepLayerNet
import time

# SimpleCG_SPD
class SimpleCG_SPD:
    def __init__(self, lr=0.01, reset_interval=20, grad_clip=5.0):
        self.lr = lr
        self.prev_dirs = {}
        self.prev_grads = {}
        self.reset_interval = reset_interval
        self.iteration = 0
        self.grad_clip = grad_clip

    def update(self, params, grads):
        for key in params:
            g = np.clip(grads[key], -self.grad_clip, self.grad_clip)
            
            if key not in self.prev_dirs or key not in self.prev_grads or \
               self.iteration % self.reset_interval == 0:
                d = -g
            else:
                g_prev = self.prev_grads[key]
                beta = np.clip(np.sum(g*g) / (np.sum(g_prev*g_prev) + 1e-8), 0.0, 0.9)
                d = -g + beta * self.prev_dirs[key]
                
                if np.sum(g * d) > 1e-8:
                    d = -g
            
            params[key] += self.lr * d
            self.prev_dirs[key] = d
            self.prev_grads[key] = g.copy()
        
        self.iteration += 1

# SGD
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]

def generate_mnist_like_data(n_samples=5000, n_test=1000):
    """MNIST風のサンプルデータを生成"""
    np.random.seed(0)
    
    # 10クラスの分類問題
    n_classes = 10
    input_size = 784
    
    # 訓練データ
    x_train = np.random.randn(n_samples, input_size).astype(np.float32) * 0.1
    # クラスごとにパターンを追加
    y_train_labels = np.random.randint(0, n_classes, n_samples)
    for i in range(n_samples):
        class_pattern = np.sin(np.arange(input_size) * y_train_labels[i] / 10.0) * 0.3
        x_train[i] += class_pattern
    
    # 正規化
    x_train = (x_train - x_train.mean()) / (x_train.std() + 1e-7)
    x_train = np.clip(x_train, 0, 1)
    
    # One-hot encoding
    t_train = np.zeros((n_samples, n_classes))
    t_train[np.arange(n_samples), y_train_labels] = 1
    
    # テストデータ
    x_test = np.random.randn(n_test, input_size).astype(np.float32) * 0.1
    y_test_labels = np.random.randint(0, n_classes, n_test)
    for i in range(n_test):
        class_pattern = np.sin(np.arange(input_size) * y_test_labels[i] / 10.0) * 0.3
        x_test[i] += class_pattern
    
    x_test = (x_test - x_test.mean()) / (x_test.std() + 1e-7)
    x_test = np.clip(x_test, 0, 1)
    
    t_test = np.zeros((n_test, n_classes))
    t_test[np.arange(n_test), y_test_labels] = 1
    
    return (x_train, t_train), (x_test, t_test)

def train_network(network, optimizer, x_train, t_train, x_test, t_test,
                  epochs=15, batch_size=100, verbose=True):
    train_size = x_train.shape[0]
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        
        epoch_loss = 0
        iter_count = 0
        
        for i in range(0, train_size, batch_size):
            batch_mask = indices[i:i+batch_size]
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)
            
            loss = network.loss(x_batch, t_batch)
            epoch_loss += loss
            iter_count += 1
        
        train_loss = epoch_loss / iter_count
        train_acc = network.accuracy(x_train[:1000], t_train[:1000])
        test_acc = network.accuracy(x_test, t_test)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc*100:5.2f}%, Test Acc: {test_acc*100:5.2f}%, "
                  f"Time: {elapsed_time:4.1f}s")
    
    total_time = time.time() - start_time
    if verbose:
        print(f"総学習時間: {total_time:.1f}秒")
    
    return train_loss_list, train_acc_list, test_acc_list

def plot_results(sgd_results, cg_results, save_path='/mnt/user-data/outputs/deep_learning_comparison.png'):
    sgd_loss, sgd_train_acc, sgd_test_acc = sgd_results
    cg_loss, cg_train_acc, cg_test_acc = cg_results
    
    epochs = len(sgd_loss)
    x = np.arange(1, epochs + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(x, sgd_loss, 'o-', label='SGD', linewidth=2, markersize=6, color='#1f77b4')
    axes[0].plot(x, cg_loss, 's-', label='SimpleCG_SPD', linewidth=2, markersize=6, color='#ff7f0e')
    axes[0].set_xlabel('Epoch', fontsize=13)
    axes[0].set_ylabel('Loss', fontsize=13)
    axes[0].set_title('(a) Loss Convergence Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Train Accuracy
    axes[1].plot(x, np.array(sgd_train_acc)*100, 'o-', label='SGD', linewidth=2, markersize=6, color='#1f77b4')
    axes[1].plot(x, np.array(cg_train_acc)*100, 's-', label='SimpleCG_SPD', linewidth=2, markersize=6, color='#ff7f0e')
    axes[1].set_xlabel('Epoch', fontsize=13)
    axes[1].set_ylabel('Accuracy (%)', fontsize=13)
    axes[1].set_title('(b) Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[2].plot(x, np.array(sgd_test_acc)*100, 'o-', label='SGD', linewidth=2, markersize=6, color='#1f77b4')
    axes[2].plot(x, np.array(cg_test_acc)*100, 's-', label='SimpleCG_SPD', linewidth=2, markersize=6, color='#ff7f0e')
    axes[2].set_xlabel('Epoch', fontsize=13)
    axes[2].set_ylabel('Accuracy (%)', fontsize=13)
    axes[2].set_title('(c) Test Accuracy', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n結果を {save_path} に保存しました")
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print(" 深層ニューラルネットワークによる最適化手法の比較実験")
    print("="*70)
    
    print("\nサンプルデータを生成中...")
    (x_train, t_train), (x_test, t_test) = generate_mnist_like_data(n_samples=5000, n_test=1000)
    
    input_size = 784
    hidden_size_list = [512, 256, 128, 64, 32]  # 5層の隠れ層（深層学習）
    output_size = 10
    epochs = 15
    batch_size = 100
    
    print(f"\n【ネットワーク構成】")
    print(f"  入力層: {input_size}ニューロン")
    for i, h in enumerate(hidden_size_list, 1):
        print(f"  隠れ層{i}: {h}ニューロン (ReLU)")
    print(f"  出力層: {output_size}ニューロン (Softmax)")
    print(f"\n  総層数: {len(hidden_size_list) + 2}層")
    print(f"  総パラメータ数: 約{(784*512 + 512*256 + 256*128 + 128*64 + 64*32 + 32*10) / 1000:.0f}K")
    print(f"\n【学習設定】")
    print(f"  訓練データ数: {len(x_train):,}")
    print(f"  テストデータ数: {len(x_test):,}")
    print(f"  エポック数: {epochs}")
    print(f"  バッチサイズ: {batch_size}")
    
    # SGD
    print("\n" + "="*70)
    print(" SGDによる学習")
    print("="*70)
    np.random.seed(42)
    network_sgd = DeepLayerNet(input_size, hidden_size_list, output_size,
                               activation='relu', weight_init_std='relu',
                               weight_decay_lambda=0.01)
    optimizer_sgd = SGD(lr=0.1)
    sgd_results = train_network(network_sgd, optimizer_sgd, x_train, t_train,
                                x_test, t_test, epochs=epochs, batch_size=batch_size)
    
    # SimpleCG_SPD
    print("\n" + "="*70)
    print(" SimpleCG_SPDによる学習")
    print("="*70)
    np.random.seed(42)
    network_cg = DeepLayerNet(input_size, hidden_size_list, output_size,
                             activation='relu', weight_init_std='relu',
                             weight_decay_lambda=0.01)
    optimizer_cg = SimpleCG_SPD(lr=0.05, reset_interval=20, grad_clip=5.0)
    cg_results = train_network(network_cg, optimizer_cg, x_train, t_train,
                               x_test, t_test, epochs=epochs, batch_size=batch_size)
    
    # 結果の比較
    print("\n" + "="*70)
    print(" 最終結果の比較")
    print("="*70)
    print(f"  SGD         - 最終Train精度: {sgd_results[1][-1]*100:5.2f}%, 最終Test精度: {sgd_results[2][-1]*100:5.2f}%")
    print(f"  SimpleCG_SPD - 最終Train精度: {cg_results[1][-1]*100:5.2f}%, 最終Test精度: {cg_results[2][-1]*100:5.2f}%")
    print(f"\n  SimpleCG_SPDの性能向上: Test精度 {(cg_results[2][-1] - sgd_results[2][-1])*100:+.2f}%")
    
    # グラフ保存
    plot_results(sgd_results, cg_results)
    
    print("\n" + "="*70)
    print(" 実験が完了しました！")
    print("="*70)
