# Para capturar os quadros
from xml.parsers.expat import model
import cv2

# Para processar o array de imagens
import numpy as np


# importe os módulos tensorflow e carregue o modelo



# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

# Loop infinito
while True:

	# Lendo / requisitando um quadro da câmera 
	status , frame = camera.read()

	# Se tivemos sucesso ao ler o quadro
	if status:

		# Inverta o quadro
		frame = cv2.flip(frame , 1)
		
		
				
		# Redimensione o quadro
		img = cv2.resize(frame,(224,224))
				# Expanda a dimensão do array junto com o eixo 0
		test_image = np.array(img, dtype=np.float32)
		test_image = np.expand_dims(test_image, axis=0)
				# Normalize para facilitar o processamento
		normalised_image = test_image/255.0
				# Obtenha previsões do modelo
		prediction = model.predict(normalised_image)
		
		# Exibindo os quadros capturados
		cv2.imshow('feed' , frame)

		# Aguardando 1ms
		code = cv2.waitKey(1)
		
		# Se a barra de espaço foi pressionada, interrompa o loop
		if code == 32:
			break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()