import torch
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
import os

torch.manual_seed(123)

classificador = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(in_features=14*14*64, out_features=128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 13)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classificador.parameters(), lr=0.001)  

transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classificador.to(device)

def training_loop(loader, epoch):
    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = classificador(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.mean((predicted == labels).float())
        running_accuracy += accuracy

        print('\rÉPOCA {:3d} - Loop {:3d} de {:3d}: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, i + 1, len(loader), loss.item(), accuracy.item()), end='\r')

    print('\rÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss / len(loader), running_accuracy / len(loader)))

def classificar_imagem(fname):
    image = Image.open(fname)
    plt.imshow(image)

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    classificador.eval()
    output = classificador(image)

    _, predicted = torch.max(output.data, 1)
    predicted_class = test_dataset.classes[predicted.item()]
    print('Previsão:', predicted_class)
    print("")

    return predicted_class

data_dir_train = 'dataset/training_set'
data_dir_test = 'dataset/test_set'

train_dataset = datasets.ImageFolder(data_dir_train, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.ImageFolder(data_dir_test, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

for epoch in range(15):
    print("")
    print('TREINAMENTO DO MODELO')
    training_loop(train_loader, epoch)
    classificador.eval()
    print("")
    print('VALIDAÇÃO DO MODELO')
    training_loop(test_loader, epoch)
    classificador.train()
    print("")
    print('----------------------------------------------------------------------------------')
print("")
print("")
import PIL.Image as Image

def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception as e:
        print(f"Erro ao carregar imagem no caminho: {path}")
        print(f"Mensagem de erro: {str(e)}")
        return None  # Você pode optar por retornar uma imagem padrão ou lidar com o erro de outra maneira

# Use este carregador em seu código de carregamento de conjunto de dados

print("-----------------------------------------------------------")

print("_____________________ART_____________________")
acertosart = 0
imagens = os.listdir('dataset/test_set/ART/0001-1000')
path = 'dataset/test_set/ART/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: ART')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "ART":
        acertosart += 1

print("-----------------------------------------------------------")

print("_____________________BLA_____________________")
acertosbla = 0
imagens = os.listdir('dataset/test_set/BLA/0001-1000')
path = 'dataset/test_set/BLA/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: BLA')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "BLA":
        acertosbla += 1

print("-----------------------------------------------------------")


print("_____________________EBO_____________________")
acertosebo = 0
imagens = os.listdir('dataset/test_set/EBO/0001-1000')
path = 'dataset/test_set/EBO/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: EBO')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "EBO":
        acertosebo += 1

print("-----------------------------------------------------------")


print("_____________________EOS_____________________")
acertoseos = 0
imagens = os.listdir('dataset/test_set/EOS/0001-1000')
path = 'dataset/test_set/EOS/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: EOS')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "EOS":
        acertoseos += 1

print("-----------------------------------------------------------")


print("_____________________LYT_____________________")
acertoslyt = 0
imagens = os.listdir('dataset/test_set/LYT/0001-1000')
path = 'dataset/test_set/LYT/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: LYT')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "LYT":
        acertoslyt += 1

print("-----------------------------------------------------------")

print("_____________________MMZ_____________________")
acertosMMZ = 0
imagens = os.listdir('dataset/test_set/MMZ/0001-1000')
path = 'dataset/test_set/MMZ/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: MMZ')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "MMZ":
        acertosMMZ += 1

print("-----------------------------------------------------------")

print("_____________________MON_____________________")
acertosMON = 0
imagens = os.listdir('dataset/test_set/MON/0001-1000')
path = 'dataset/test_set/MON/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: MON')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "MON":
        acertosMON += 1

print("-----------------------------------------------------------")

print("_____________________NGB_____________________")
acertosNGB = 0
imagens = os.listdir('dataset/test_set/NGB/0001-1000')
path = 'dataset/test_set/NGB/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: NGB')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "NGB":
        acertosNGB += 1

print("-----------------------------------------------------------")


print("_____________________NGS_____________________")
acertosNGS = 0
imagens = os.listdir('dataset/test_set/NGS/0001-1000')
path = 'dataset/test_set/NGS/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: NGS')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "NGS":
        acertosNGS += 1

print("-----------------------------------------------------------")
print("")

print("_____________________NIF_____________________")
acertosNIF = 0
imagens = os.listdir('dataset/test_set/NIF/0001-1000')
path = 'dataset/test_set/NIF/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: NIF')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "NIF":
        acertosNIF += 1

print("-----------------------------------------------------------")

print("_____________________PEB_____________________")
acertosPEB = 0
imagens = os.listdir('dataset/test_set/PEB/0001-1000')
path = 'dataset/test_set/PEB/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: PEB')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "PEB":
        acertosPEB += 1

print("-----------------------------------------------------------")


print("_____________________PLM_____________________")
acertosPLM = 0
imagens = os.listdir('dataset/test_set/PLM/0001-1000')
path = 'dataset/test_set/PLM/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: PLM')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "PLM":
        acertosPLM += 1

print("-----------------------------------------------------------")


print("_____________________PMO_____________________")
acertosPMO = 0
imagens = os.listdir('dataset/test_set/PMO/0001-1000')
path = 'dataset/test_set/PMO/0001-1000'
for imagem in imagens:
    print('Tipo de celula esperada: PMO')
    fullpath = path+'/'+imagem
    result = classificar_imagem(fullpath)
    if result == "PMO":
        acertosPMO += 1

print("-----------------------------------------------------------")

print("")
print(f'Taxa de acertos do ART: {(acertosart/len(imagens))*100}%')
print(f'Taxa de acertos do BLA: {(acertosbla/len(imagens))*100}%')
print(f'Taxa de acertos do EBO: {(acertosebo/len(imagens))*100}%')
print(f'Taxa de acertos do EOS: {(acertoseos/len(imagens))*100}%')
print(f'Taxa de acertos do LYT: {(acertoslyt/len(imagens))*100}%')
print(f'Taxa de acertos do MMZ: {(acertosMMZ/len(imagens))*100}%')
print(f'Taxa de acertos do MON: {(acertosMON/len(imagens))*100}%')
print(f'Taxa de acertos do NGB: {(acertosNGB/len(imagens))*100}%')
print(f'Taxa de acertos do NGS: {(acertosNGS/len(imagens))*100}%')
print(f'Taxa de acertos do NIF: {(acertosNIF/len(imagens))*100}%')
print(f'Taxa de acertos do PEB: {(acertosPEB/len(imagens))*100}%')
print(f'Taxa de acertos do PLM: {(acertosPLM/len(imagens))*100}%')
print(f'Taxa de acertos do PMO: {(acertosPMO/len(imagens))*100}%')

# Corrija o nome do modelo para 'classificador' e especifique o caminho correto para salvar o modelo
torch.save(classificador.state_dict(), 'dataset/classificador.pth')
