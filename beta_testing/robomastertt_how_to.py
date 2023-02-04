import time
import tellopy


drone = tellopy.Tello()
drone.connect()
print("Enviando comandos")
time.sleep(3)

"""
El patron de la pantalla es una matriz de 8x8 
para mandar un patron se mandan los 64 recuadros seguidos en orden de izquierda a derecha y de arriba hacia abajo:

Ejemplo: Si se quiere mandar solo la primera fila iluminada de rojo, el patron seria: 
rrrrrrrr00000000000000000000000000000000000000000000000000000000

Si se quiere solo la ultima fila en lila seria:
00000000000000000000000000000000000000000000000000000000pppppppp

los colores posibles creo que son solo r(rojo), b(azul), p(lila)

"""


# Corazon
drone.send_packet_data("EXT mled g 000000000rr00rr0rrrrrrrrrrrrrrrr0rrrrrr000rrrr00000rr00000000000")
time.sleep(3)

# Sonrisa
drone.send_packet_data("EXT mled g 00pppp000p0000p0p0p00p0pp000000pp0p00p0pp00pp00p0p0000p000pppp00")
time.sleep(3)

# Cara triste
drone.send_packet_data("EXT mled g 00pppp000p0000p0p0p00p0pp000000pp00pp00pp0p00p0p0p0000p000pppp00")
time.sleep(3)

# Escribir texto
drone.send_packet_data("EXT mled l r 2.5 HOLA")
time.sleep(6)

# Pantalla moviendose con un patron (hacia arriba)
drone.send_packet_data("EXT mled u g 2.5 0000b00bbb0b0b000b00b00000bb0000000b0000bbb00bbb000b0b0b0b00b0b0")
time.sleep(3)

"""

Otros comandos para mandar en "send_packet_data":

EXT mled s p I   # write a letter
EXT mled sg 0000000000000000000000000000000000000000000000000000000000000000    # make a starting pattern
EXT led 255 255 255  # turn on top led

"""

