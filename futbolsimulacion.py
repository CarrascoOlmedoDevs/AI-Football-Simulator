import pygame
import random
import math
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque

# Inicializar Pygame
pygame.init()

# Configuración de la pantalla
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simulación Avanzada de Fútbol IA vs IA")

# Colores
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Dimensiones del campo
FIELD_WIDTH = 700
FIELD_HEIGHT = 500
FIELD_START_X = (SCREEN_WIDTH - FIELD_WIDTH) // 2
FIELD_START_Y = (SCREEN_HEIGHT - FIELD_HEIGHT) // 2

# --- Enumeraciones ---
class EstadoJugador(Enum):
    INACTIVO = 1
    PERSEGUIR_BALON = 2
    CON_BALON = 3
    APOYAR_ATAQUE = 4
    DEFENDER = 5
    VOLVER_A_POSICION = 6
    LESIONADO = 7

class EfectoClima(Enum):
    NORMAL = 1
    LLUVIA = 2
    VIENTO = 3

# --- Sistema de Memoria del Jugador ---
class MemoriaJugador:
    def __init__(self, capacidad=1000):
        self.capacidad = capacidad
        self.memoria = deque(maxlen=capacidad)

    def agregar_experiencia(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.memoria.append((estado, accion, recompensa, siguiente_estado, terminado))

    def obtener_batch(self, batch_size):
        return random.sample(self.memoria, min(len(self.memoria), batch_size))

# --- Red Neuronal para DQN ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
from enum import Enum

class EstadoJugador(Enum):
    INACTIVO = 1
    PERSEGUIR_BALON = 2
    CON_BALON = 3
    APOYAR_ATAQUE = 4
    DEFENDER = 5
    VOLVER_A_POSICION = 6
    LESIONADO = 7

class MemoriaJugador:
    def __init__(self, capacidad=10000):
        self.capacidad = capacidad
        self.memoria = []
        self.posicion = 0

    def agregar(self, estado, accion, recompensa, siguiente_estado, terminado):
        if len(self.memoria) < self.capacidad:
            self.memoria.append(None)
        self.memoria[self.posicion] = (estado, accion, recompensa, siguiente_estado, terminado)
        self.posicion = (self.posicion + 1) % self.capacidad

    def muestra(self, batch_size):
        return random.sample(self.memoria, min(len(self.memoria), batch_size))

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class EstadoJugador(Enum):
    INACTIVO = 1
    PERSEGUIR_BALON = 2
    CON_BALON = 3
    APOYAR_ATAQUE = 4
    DEFENDER = 5
    VOLVER_A_POSICION = 6
    LESIONADO = 7

class JugadorIA(pygame.sprite.Sprite):
    def __init__(self, x, y, color, rol, equipo_lado, juego, ofensivo=False):
        super().__init__()
        self.juego = juego
        self.tamano_celda = 50
        self.ofensivo = ofensivo
        self.tiempo_con_balon = 0
        self.porteria_propia = None
        self.porteria_oponente = None
        # Atributos específicos de IA
        self.estado_size = 28
        self.accion_size = 6
        self.memoria = MemoriaJugador(10000)
        self.modelo = DQN(self.estado_size, self.accion_size)
        self.modelo_objetivo = DQN(self.estado_size, self.accion_size)
        self.optimizador = optim.Adam(self.modelo.parameters(), lr=0.001)
        self.epsilon = 0.1
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 100
        self.target_update_counter = 0
        self.goles_en_contra = 0

        self.image = pygame.Surface((20, 20))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        self.rol = rol
        self.lado = equipo_lado
        self.estado = EstadoJugador.INACTIVO
        self.balon = None
        self.equipo = None
        self.posicion_base = (x, y)
        self.objetivo = None

        # Atributos mejorados
        self.velocidad = self.generar_atributo(rol, 'velocidad')
        self.resistencia = self.generar_atributo(rol, 'resistencia')
        self.pase = self.generar_atributo(rol, 'pase')
        self.tiro = self.generar_atributo(rol, 'tiro')
        self.regate = self.generar_atributo(rol, 'regate')
        self.entrada = self.generar_atributo(rol, 'entrada')
        self.porteria = self.generar_atributo(rol, 'porteria')
        self.vision = self.generar_atributo(rol, 'vision')
        self.posicionamiento = self.generar_atributo(rol, 'posicionamiento')

        self.fatiga = 0
        self.propension_lesion = random.randint(1, 10)
        self.lesionado = False
        self.tiempo_lesion = 0

        self.tarjetas_amarillas = 0
        self.tarjeta_roja = False

        self.velocidad_vector = pygame.math.Vector2(0, 0)
        self.velocidad_maxima = 8
        self.aceleracion = 16
        self.friccion = 0.1

    def generar_atributo(self, rol, atributo):
        valor_base = random.randint(50, 80)
        if rol == 'PO' and atributo in ['porteria', 'posicionamiento']:
            valor_base += 20
        elif rol == 'DE' and atributo in ['entrada', 'posicionamiento']:
            valor_base += 15
        elif rol == 'MC' and atributo in ['pase', 'vision']:
            valor_base += 15
        elif rol == 'DL' and atributo in ['tiro', 'regate']:
            valor_base += 15
        return min(valor_base, 100)

    def obtener_estado(self):
        if not self.balon or not self.equipo or not self.equipo.oponente:
            return torch.zeros(1, self.estado_size)

        estado = [
            self.rect.centerx / self.juego.FIELD_WIDTH,
            self.rect.centery / self.juego.FIELD_HEIGHT,
            self.balon.rect.centerx / self.juego.FIELD_WIDTH,
            self.balon.rect.centery / self.juego.FIELD_HEIGHT,
            int(self.balon.poseedor == self),
            int(self.equipo.en_posesion),
            self.resistencia / 100,
            self.fatiga / 100,
            int(self.estado == EstadoJugador.INACTIVO),
            self.pase / 100,
            self.tiro / 100,
            self.regate / 100,
            self.entrada / 100,
            self.velocidad / 100,
            int(self.lesionado),
            self.tarjetas_amarillas,
            int(self.tarjeta_roja),
            self.goles_en_contra / 10,
            self.equipo.goles_en_contra / 10,
        ]

        # Añadir información sobre jugadores cercanos
        jugadores_cercanos = self.obtener_jugadores_cercanos()
        for jugador in jugadores_cercanos[:3]:  # Considerar los 3 jugadores más cercanos
            estado.extend([
                (jugador.rect.centerx - self.rect.centerx) / self.juego.FIELD_WIDTH,
                (jugador.rect.centery - self.rect.centery) / self.juego.FIELD_HEIGHT,
                int(jugador.equipo == self.equipo)
            ])

        # Rellenar con ceros si hay menos de 3 jugadores cercanos
        estado.extend([0] * (3 - len(jugadores_cercanos)) * 3)

        # Asegúrate de que el estado tenga exactamente self.estado_size elementos
        assert len(estado) == self.estado_size, f"El tamaño del estado ({len(estado)}) no coincide con self.estado_size ({self.estado_size})"

        return torch.tensor(estado, dtype=torch.float32).view(1, -1)

    def elegir_accion(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.accion_size - 1)
        else:
            estado = self.obtener_estado()
            with torch.no_grad():
                q_values = self.modelo(estado)
            return torch.argmax(q_values).item()

    def entrenar(self):
        if len(self.memoria.memoria) < self.batch_size:
            return

        batch = self.memoria.muestra(self.batch_size)
        estados = torch.cat([exp[0] for exp in batch])
        acciones = torch.tensor([exp[1] for exp in batch], dtype=torch.long).unsqueeze(1)
        recompensas = torch.tensor([exp[2] for exp in batch], dtype=torch.float32).unsqueeze(1)
        siguientes_estados = torch.cat([exp[3] for exp in batch])
        terminados = torch.tensor([exp[4] for exp in batch], dtype=torch.float32).unsqueeze(1)

        q_valores_actuales = self.modelo(estados).gather(1, acciones)
        q_valores_siguientes = self.modelo_objetivo(siguientes_estados).max(1)[0].unsqueeze(1)
        q_valores_objetivo = recompensas + (1 - terminados) * self.gamma * q_valores_siguientes

        loss = nn.MSELoss()(q_valores_actuales, q_valores_objetivo)
        self.optimizador.zero_grad()
        loss.backward()
        self.optimizador.step()

        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.modelo_objetivo.load_state_dict(self.modelo.state_dict())
            self.target_update_counter = 0

    def actualizar(self):
        if self.lesionado:
            self.manejar_lesion()
            return

        estado_actual = self.obtener_estado()
        accion = self.elegir_accion()
        recompensa = self.ejecutar_accion(accion)
        siguiente_estado = self.obtener_estado()
        terminado = self.juego.tiempo >= 90 * 60

        self.memoria.agregar(estado_actual, accion, recompensa, siguiente_estado, terminado)
        self.entrenar()

        self.actualizar_resistencia()
        self.actualizar_fatiga()
        self.comprobar_lesion()
        # Añadir esta comprobación
        if self.distancia_a(self.balon.rect.center) < 100 and self.balon.poseedor is None:
            self.mover_hacia(self.balon.rect.center)
        if self.objetivo:
            self.mover_hacia(self.objetivo)
        if self.balon.poseedor == self:
            self.tiempo_con_balon += 1 / 60  # Assuming 60 FPS
            if self.tiempo_con_balon > 10:
                self.penalizar_por_tiempo_excesivo()
        else:
            self.tiempo_con_balon = 0
        if self.estado == EstadoJugador.CON_BALON:
            distancia_a_porteria_oponente = self.distancia_a(self.porteria_oponente.center)
            if distancia_a_porteria_oponente < 200 and random.random() < 0.7:
                self.tirar()
    def penalizar_por_tiempo_excesivo(self):
        self.puntuacion -= 5  # Reduce player's score
        self.balon.liberar()  # Force the player to release the ball
        print(f"{self.rol} penalizado por retener el balón demasiado tiempo")
    def ejecutar_accion(self, accion):
        self.ultima_accion = accion

        recompensa = 0
        if accion == 0:  # Mover
            self.mover_hacia(self.balon.rect.center)
            recompensa += 1
        elif accion == 1:  # Pasar
            if self.balon.poseedor == self:
                self.pasar_balon()
                recompensa += 5
        elif accion == 2:  # Tirar
            if self.balon.poseedor == self:
                self.tirar()
                recompensa += 10
        elif accion == 3:  # Entrar
            recompensa += self.intentar_entrada()
        elif accion == 4:  # Regatear
            if self.balon.poseedor == self:
                self.regatear()
                recompensa += 2
        elif accion == 5:  # Esperar
            recompensa -= 1

        return recompensa

    def mover_hacia(self, objetivo):
        direccion = pygame.math.Vector2(objetivo) - pygame.math.Vector2(self.rect.center)
        if direccion.length() > 5:  # Cambiamos esto para que se acerquen más
            direccion = direccion.normalize()
            self.velocidad_vector += direccion * self.aceleracion
            if self.velocidad_vector.length() > self.velocidad_maxima:
                self.velocidad_vector.scale_to_length(self.velocidad_maxima)
        
        nueva_pos = pygame.math.Vector2(self.rect.center) + self.velocidad_vector
        self.rect.center = nueva_pos
        self.velocidad_vector *= self.friccion
    def pasar_balon(self):
        if self.balon.poseedor != self:
            return
        
        companero_mas_cercano = min(
            [j for j in self.equipo.jugadores if j != self],
            key=lambda j: self.distancia_a(j.rect.center)
        )
        
        direccion = pygame.math.Vector2(companero_mas_cercano.rect.center) - pygame.math.Vector2(self.rect.center)
        if direccion.length() > 0:
            direccion = direccion.normalize()
        
        fuerza_pase = self.pase * 0.5
        self.balon.patear(fuerza_pase, direccion)
        print(f"{self.rol} pasó el balón")

    def tirar(self):
        if self.balon.poseedor != self:
            return
        
        porteria_x = self.juego.FIELD_START_X + self.juego.FIELD_WIDTH if self.lado == 'izquierda' else self.juego.FIELD_START_X
        porteria_y = self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2
        
        direccion = pygame.math.Vector2(porteria_x, porteria_y) - pygame.math.Vector2(self.rect.center)
        if direccion.length() > 0:
            direccion = direccion.normalize()
        
        fuerza_tiro = self.tiro * 0.7
        self.balon.patear(fuerza_tiro, direccion)
        print(f"{self.rol} tiró a portería")

    def intentar_entrada(self):
        oponente_cercano = min(
            self.equipo.oponente.jugadores,
            key=lambda j: self.distancia_a(j.rect.center)
        )
        
        if self.distancia_a(oponente_cercano.rect.center) < 30:
            exito = random.random() < (self.entrada / 100)
            if exito:
                self.balon.poseedor = None
                print(f"{self.rol} realizó una entrada exitosa")
                return 5
            else:
                print(f"{self.rol} falló en la entrada")
                return -2
        return 0

    def regatear(self):
        if self.balon.poseedor != self:
            return
        
        direccion = pygame.math.Vector2(
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ).normalize()
        
        self.mover_hacia(self.rect.center + direccion * 20)
        self.balon.rect.center = self.rect.center

    def actualizar_resistencia(self):
        if self.velocidad_vector.length() > 0:
            self.resistencia = max(0, self.resistencia - 0.1)
        else:
            self.resistencia = min(100, self.resistencia + 0.05)

    def actualizar_fatiga(self):
        self.fatiga = min(100, self.fatiga + 0.01)
        self.velocidad_maxima = self.velocidad * (1 - self.fatiga / 200)

    def manejar_lesion(self):
        self.tiempo_lesion -= 1
        if self.tiempo_lesion <= 0:
            self.lesionado = False
            self.estado = EstadoJugador.INACTIVO

    def comprobar_lesion(self):
        if random.random() < 0.00000001 * self.propension_lesion:
            self.lesionado = True
            self.tiempo_lesion = random.randint(300, 1800)
            self.estado = EstadoJugador.LESIONADO
            print(f"{self.rol} se ha lesionado")

    def obtener_jugadores_cercanos(self):
        return sorted(
            [j for j in self.juego.equipo1.jugadores + self.juego.equipo2.jugadores if j != self],
            key=lambda j: self.distancia_a(j.rect.center)
        )[:5]

    def distancia_a(self, punto):
        return math.hypot(self.rect.centerx - punto[0], self.rect.centery - punto[1])

    def dibujar(self, pantalla):
        pygame.draw.circle(pantalla, self.image.get_at((0, 0)), self.rect.center, 10)
        fuente = pygame.font.Font(None, 20)
        texto = fuente.render(self.rol[0], True, (255, 255, 255))
        pantalla.blit(texto, (self.rect.centerx - 5, self.rect.centery - 5))

        if self.lesionado:
            pygame.draw.circle(pantalla, (255, 0, 0), (self.rect.centerx, self.rect.centery - 15), 5)

    def guardar_modelo(self, ruta_archivo):
        torch.save(self.modelo.state_dict(), ruta_archivo)
        print(f"Modelo guardado en {ruta_archivo}")

    def cargar_modelo(self, ruta_archivo):
        if os.path.exists(ruta_archivo):
            self.modelo.load_state_dict(torch.load(ruta_archivo))
            self.modelo.eval()
            print(f"Modelo cargado desde {ruta_archivo}")
        else:
            print(f"No se encontró el archivo {ruta_archivo}, iniciando nuevo modelo")

    def reiniciar(self):
        self.fatiga = 0
        self.lesionado = False
        self.tiempo_lesion = 0
        self.estado = EstadoJugador.INACTIVO
        self.velocidad_vector = pygame.math.Vector2(0, 0)
        self.rect.center = self.posicion_base
        self.objetivo = None

    def es_el_mas_cercano_al_balon(self):
        return self == min(self.equipo.jugadores, key=lambda p: p.distancia_a(self.balon.rect.center))

    def considerar_perseguir_balon(self):
        if self.es_el_mas_cercano_al_balon() and self.distancia_a(self.balon.rect.center) < 150:
            self.estado = EstadoJugador.PERSEGUIR_BALON

    def estado_inactivo(self):
        if self.balon.poseedor is None and self.distancia_a(self.balon.rect.center) < 100:
            self.estado = EstadoJugador.PERSEGUIR_BALON
        elif self.equipo.en_posesion and self.rol in ['DE', 'MC']:
            self.estado = EstadoJugador.APOYAR_ATAQUE
        elif not self.equipo.en_posesion:
            self.estado = EstadoJugador.DEFENDER
        else:
            self.mover_hacia(self.posicion_base)

    def estado_perseguir_balon(self):
        if self.balon.poseedor is None:
            self.mover_hacia(self.balon.rect.center)
            if self.rect.colliderect(self.balon.rect):
                self.estado = EstadoJugador.CON_BALON
                self.balon.poseedor = self
                print(f"Jugador {self.rol} del equipo {self.equipo.lado} ha recogido el balón")
        else:
            self.estado = EstadoJugador.INACTIVO

    def estado_con_balon(self):
        if self.balon.poseedor != self:
            self.estado = EstadoJugador.INACTIVO
        else:
            if self.rol == 'PO':
                self.portero_con_balon()
            elif self.rol == 'DE':
                self.defensor_con_balon()
            else:
                if self.bajo_presion():
                    self.pasar_balon()
                elif self.deberia_tirar():
                    self.tirar()
                else:
                    self.regatear()

    def estado_apoyar_ataque(self):
        if not self.equipo.en_posesion:
            self.estado = EstadoJugador.DEFENDER
        else:
            objetivo = self.encontrar_espacio_libre()
            self.mover_hacia(objetivo)

    def estado_defender(self):
        if self.equipo.en_posesion:
            self.estado = EstadoJugador.VOLVER_A_POSICION
        elif self.rol == 'PO':
            self.portero_defender()
        elif self.rol == 'DE':
            self.acciones_defensor()
        else:
            oponente_cercano = self.encontrar_oponente_mas_cercano()
            if oponente_cercano:
                self.mover_hacia(oponente_cercano.rect.center)
            else:
                self.mover_hacia(self.posicion_base)

    def estado_volver_a_posicion(self):
        self.mover_hacia(self.posicion_base)
        if self.distancia_a(self.posicion_base) < 10:
            self.estado = EstadoJugador.INACTIVO

    def portero_con_balon(self):
        if random.random() < 0.7:
            self.pasar_balon()
        else:
            self.despejar_balon()

    def portero_defender(self):
        centro_porteria = (self.juego.FIELD_START_X, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2) if self.equipo.lado == 'izquierda' else (self.juego.FIELD_START_X + self.juego.FIELD_WIDTH, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2)
        balon_a_porteria = pygame.math.Vector2(centro_porteria) - pygame.math.Vector2(self.balon.rect.center)
        punto_intercepcion = pygame.math.Vector2(self.balon.rect.center) + balon_a_porteria * 0.3
        self.mover_hacia(punto_intercepcion)

    def defensor_con_balon(self):
        if self.bajo_presion():
            if random.random() < 0.7:
                self.despejar_balon()
            else:
                self.pasar_balon()
        else:
            if random.random() < 0.8:
                self.pasar_balon()
            else:
                self.regatear()

    def acciones_defensor(self):
        oponente_cercano = self.encontrar_oponente_mas_cercano()
        if oponente_cercano:
            distancia_oponente = self.distancia_a(oponente_cercano.rect.center)
            if distancia_oponente < 30 and oponente_cercano.balon and oponente_cercano.balon.poseedor == oponente_cercano:
                self.intentar_entrada()
            elif distancia_oponente < 100:
                self.mover_hacia(oponente_cercano.rect.center)
            else:
                self.posicionamiento_defensivo()
        else:
            self.posicionamiento_defensivo()

    def bajo_presion(self):
        oponentes_cercanos = [oponente for oponente in self.equipo.oponente.jugadores
                              if self.distancia_a(oponente.rect.center) < 50]
        return len(oponentes_cercanos) >= 1

    def deberia_tirar(self):
        centro_porteria = (self.juego.FIELD_START_X + self.juego.FIELD_WIDTH, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2) if self.equipo.lado == 'izquierda' else (self.juego.FIELD_START_X, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2)
        distancia_porteria = self.distancia_a(centro_porteria)
        rango_tiro = self.juego.FIELD_WIDTH * 0.4

        if distancia_porteria <= rango_tiro:
            probabilidad_tiro = (1 - distancia_porteria / rango_tiro) * (self.tiro / 100) * 1.5 
            return random.random() < probabilidad_tiro
        return False

    def despejar_balon(self):
        direccion = pygame.math.Vector2(1, 0) if self.equipo.lado == 'izquierda' else pygame.math.Vector2(-1, 0)
        direccion.rotate_ip(random.uniform(-30, 30))
        potencia = self.pase * 3
        
        self.balon.patear(potencia, direccion)
        self.balon.poseedor = None
        print(f"Balón despejado por {self.rol}")

    def posicionamiento_defensivo(self):
        centro_porteria = (self.juego.FIELD_START_X, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2) if self.equipo.lado == 'izquierda' else (self.juego.FIELD_START_X + self.juego.FIELD_WIDTH, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2)
        balon_a_porteria = pygame.math.Vector2(centro_porteria) - pygame.math.Vector2(self.balon.rect.center)
        punto_intercepcion = pygame.math.Vector2(self.balon.rect.center) + balon_a_porteria * 0.3
        self.mover_hacia(punto_intercepcion)

    def encontrar_espacio_libre(self):
        for _ in range(10):
            x = random.randint(self.juego.FIELD_START_X, self.juego.FIELD_START_X + self.juego.FIELD_WIDTH)
            y = random.randint(self.juego.FIELD_START_Y, self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT)
            if all(self.distancia_a((x, y)) > 50 for jugador in self.juego.equipo1.jugadores + self.juego.equipo2.jugadores if jugador != self):
                return (x, y)
        return self.posicion_base

    def encontrar_oponente_mas_cercano(self):
        return min(self.equipo.oponente.jugadores, key=lambda p: self.distancia_a(p.rect.center))





class Equipo:
    def __init__(self, color, lado, juego):
        self.color = color
        self.lado = lado
        self.juego = juego
        self.jugadores = []
        self.formacion = '4-3-3'
        self.en_posesion = False
        self.estado = 'neutral'
        self.oponente = None
        self.posiciones_base = self.calcular_posiciones_base()
        self.puntuacion = 0
        self.faltas = 0
        self.tarjetas_amarillas = 0
        self.tarjetas_rojas = 0
        self.suplentes = []
        self.max_sustituciones = 3
        self.sustituciones_realizadas = 0
        self.crear_jugadores()
        self.porteria_propia = None
        self.porteria_oponente = None
        self.establecer_porterias()
        self.goles_en_contra = 0

    def recibir_penalizacion_por_gol(self):
        self.goles_en_contra += 1
        # Puedes añadir más lógica aquí si es necesario


    def establecer_porterias(self):
        OFFSET_PORTERIA = 20
        PORTERIA_ALTURA = 150
        if self.lado == 'izquierda':
            self.porteria_propia = pygame.Rect(
                self.juego.FIELD_START_X - OFFSET_PORTERIA,
                self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2 - PORTERIA_ALTURA // 2,
                10,  # Ancho de la portería
                PORTERIA_ALTURA
            )
            self.porteria_oponente = pygame.Rect(
                self.juego.FIELD_START_X + self.juego.FIELD_WIDTH + OFFSET_PORTERIA,
                self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2 - PORTERIA_ALTURA // 2,
                10,
                PORTERIA_ALTURA
            )
        else:  # 'derecha'
            self.porteria_propia = pygame.Rect(
                self.juego.FIELD_START_X + self.juego.FIELD_WIDTH + OFFSET_PORTERIA,
                self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2 - PORTERIA_ALTURA // 2,
                10,
                PORTERIA_ALTURA
            )
            self.porteria_oponente = pygame.Rect(
                self.juego.FIELD_START_X - OFFSET_PORTERIA,
                self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT // 2 - PORTERIA_ALTURA // 2,
                10,
                PORTERIA_ALTURA
            )
    def calcular_posiciones_base(self):
        FIELD_START_X = self.juego.FIELD_START_X
        FIELD_START_Y = self.juego.FIELD_START_Y
        FIELD_WIDTH = self.juego.FIELD_WIDTH
        FIELD_HEIGHT = self.juego.FIELD_HEIGHT

        if self.lado == 'izquierda':
            return {
                'PO': [(FIELD_START_X + 20, FIELD_START_Y + FIELD_HEIGHT // 2)],
                'DE': [(FIELD_START_X + FIELD_WIDTH * 0.2, FIELD_START_Y + FIELD_HEIGHT * i / 5) for i in range(1, 5)],
                'MC': [(FIELD_START_X + FIELD_WIDTH * 0.4, FIELD_START_Y + FIELD_HEIGHT * i / 4) for i in range(1, 4)],
                'DL': [(FIELD_START_X + FIELD_WIDTH * 0.7, FIELD_START_Y + FIELD_HEIGHT * i / 4) for i in range(1, 4)],
                'BANDA_IZQ': [(FIELD_START_X, FIELD_START_Y + i * FIELD_HEIGHT // 4) for i in range(5)],
                'BANDA_DER': [(FIELD_START_X + FIELD_WIDTH, FIELD_START_Y + i * FIELD_HEIGHT // 4) for i in range(5)],
            }
        else:
            return {
                'PO': [(FIELD_START_X + FIELD_WIDTH - 20, FIELD_START_Y + FIELD_HEIGHT // 2)],
                'DE': [(FIELD_START_X + FIELD_WIDTH * 0.8, FIELD_START_Y + FIELD_HEIGHT * i / 5) for i in range(1, 5)],
                'MC': [(FIELD_START_X + FIELD_WIDTH * 0.6, FIELD_START_Y + FIELD_HEIGHT * i / 4) for i in range(1, 4)],
                'DL': [(FIELD_START_X + FIELD_WIDTH * 0.3, FIELD_START_Y + FIELD_HEIGHT * i / 4) for i in range(1, 4)],
                'BANDA_IZQ': [(FIELD_START_X, FIELD_START_Y + i * FIELD_HEIGHT // 4) for i in range(5)],
                'BANDA_DER': [(FIELD_START_X + FIELD_WIDTH, FIELD_START_Y + i * FIELD_HEIGHT // 4) for i in range(5)],
            }

    def crear_jugadores(self):
        formaciones = {
            '4-3-3': [('PO', 1), ('DE', 4), ('MC', 3), ('DL', 3)]
        }

        posiciones = formaciones[self.formacion]

        for rol, cantidad in posiciones:
            posiciones_disponibles = self.posiciones_base[rol]
            for i in range(min(cantidad, len(posiciones_disponibles))):
                x, y = posiciones_disponibles[i]
                ofensivo = (rol == 'MC' and i < 2)  # Marca los primeros dos MC como ofensivos
                jugador = JugadorIA(x, y, self.color, rol, self.lado, self.juego, ofensivo=ofensivo)
                jugador.equipo = self
                self.jugadores.append(jugador)

    def actualizar(self, balon):
        self.actualizar_estado_equipo(balon)
        self.ajustar_formacion()
        for jugador in self.jugadores:
            jugador.balon = balon
            jugador.actualizar()
        for jugador in self.jugadores:
            jugador.porteria_propia = self.porteria_propia
            jugador.porteria_oponente = self.porteria_oponente
        self.comprobar_sustituciones()

    def actualizar_estado_equipo(self, balon):
        self.en_posesion = balon.poseedor in self.jugadores
        
        if self.en_posesion and isinstance(balon.poseedor, JugadorIA) and balon.poseedor.rol == 'MC':
            self.estado = 'atacando'
        elif self.en_posesion:
            self.estado = 'neutral'
        elif balon.poseedor in self.oponente.jugadores:
            self.estado = 'defendiendo'
        else:
            self.estado = 'neutral'

    def ajustar_formacion(self):
        if self.estado == 'atacando':
            self.establecer_formacion_ataque()
        elif self.estado == 'defendiendo':
            self.establecer_formacion_defensa()
        else:
            self.establecer_formacion_neutral()

    def establecer_formacion_ataque(self):
        desplazamiento = 0.1 if self.lado == 'izquierda' else -0.1
        for jugador in self.jugadores:
            base_x, base_y = self.posiciones_base[jugador.rol][self.jugadores.index(jugador) % len(self.posiciones_base[jugador.rol])]
            nuevo_x = min(max(base_x + self.juego.FIELD_WIDTH * desplazamiento, self.juego.FIELD_START_X), self.juego.FIELD_START_X + self.juego.FIELD_WIDTH)
            jugador.objetivo = (nuevo_x, base_y)

    def establecer_formacion_defensa(self):
        desplazamiento = -0.05 if self.lado == 'izquierda' else 0.05
        for jugador in self.jugadores:
            base_x, base_y = self.posiciones_base[jugador.rol][self.jugadores.index(jugador) % len(self.posiciones_base[jugador.rol])]
            nuevo_x = min(max(base_x + self.juego.FIELD_WIDTH * desplazamiento, self.juego.FIELD_START_X), self.juego.FIELD_START_X + self.juego.FIELD_WIDTH)
            jugador.objetivo = (nuevo_x, base_y)

    def establecer_formacion_neutral(self):
        for jugador in self.jugadores:
            jugador.objetivo = self.posiciones_base[jugador.rol][self.jugadores.index(jugador) % len(self.posiciones_base[jugador.rol])]

    def comprobar_sustituciones(self):
        if self.sustituciones_realizadas >= self.max_sustituciones:
            return

        for jugador in self.jugadores:
            if jugador.lesionado or jugador.fatiga > 90:
                self.realizar_sustitucion(jugador)
                break

    def realizar_sustitucion(self, jugador_saliente):
        suplentes_adecuados = [sup for sup in self.suplentes if sup.rol == jugador_saliente.rol]
        if suplentes_adecuados:
            jugador_entrante = random.choice(suplentes_adecuados)
            self.jugadores.remove(jugador_saliente)
            self.jugadores.append(jugador_entrante)
            self.suplentes.remove(jugador_entrante)
            self.suplentes.append(jugador_saliente)

            jugador_entrante.rect.center = jugador_saliente.rect.center
            jugador_entrante.posicion_base = jugador_saliente.posicion_base
            self.sustituciones_realizadas += 1
            print(f"Sustitución: {jugador_saliente.rol} sale, {jugador_entrante.rol} entra")

    def dibujar(self, pantalla):
        for jugador in self.jugadores:
            jugador.dibujar(pantalla)

    def reiniciar_posiciones(self):
        for jugador in self.jugadores:
            indice_rol = self.jugadores.index(jugador) % len(self.posiciones_base[jugador.rol])
            jugador.rect.center = self.posiciones_base[jugador.rol][indice_rol]
            jugador.posicion_base = jugador.rect.center
            jugador.velocidad_vector = pygame.math.Vector2(0, 0)
            jugador.objetivo = None
            jugador.estado = EstadoJugador.INACTIVO

class Arbitro:
    def __init__(self, juego):
        self.juego = juego
        self.conteo_faltas = 0

    def actualizar(self):
        self.comprobar_fuera_de_juego()

    def comprobar_fuera_de_juego(self):
        # Implementación básica de la regla de fuera de juego
        for equipo in [self.juego.equipo1, self.juego.equipo2]:
            jugadores_atacantes = [j for j in equipo.jugadores if j.rol in ['DL', 'MC']]
            jugadores_defensores = [j for j in equipo.oponente.jugadores if j.rol in ['DE', 'PO']]
            
            for atacante in jugadores_atacantes:
                if self.esta_en_fuera_de_juego(atacante, jugadores_defensores):
                    print(f"Jugador {atacante.rol} del equipo {equipo.lado} en posición de fuera de juego")
                    # Aquí puedes agregar lógica adicional para manejar el fuera de juego
                    # Por ejemplo, detener el juego, cambiar la posesión, etc.

    def esta_en_fuera_de_juego(self, atacante, defensores):
        # Implementación simplificada de la regla de fuera de juego
        if atacante.equipo.lado == 'izquierda':
            linea_defensiva = max(defensor.rect.left for defensor in defensores)
            return atacante.rect.left > linea_defensiva and atacante.rect.left > self.juego.balon.rect.left
        else:
            linea_defensiva = min(defensor.rect.right for defensor in defensores)
            return atacante.rect.right < linea_defensiva and atacante.rect.right < self.juego.balon.rect.right

    def pitar_falta(self, jugador_infractor, jugador_afectado):
        jugador_infractor.equipo.faltas += 1
        self.conteo_faltas += 1
        print(f"Falta cometida por {jugador_infractor.rol} sobre {jugador_afectado.rol}")

        # Sistema de tarjetas
        if random.random() < 0.2:  # 20% de probabilidad de tarjeta amarilla
            self.mostrar_tarjeta_amarilla(jugador_infractor)
        
        self.otorgar_tiro_libre(jugador_afectado)

    def mostrar_tarjeta_amarilla(self, jugador):
        jugador.tarjetas_amarillas += 1
        print(f"Tarjeta amarilla para {jugador.rol}")
        if jugador.tarjetas_amarillas == 2:
            self.mostrar_tarjeta_roja(jugador)

    def mostrar_tarjeta_roja(self, jugador):
        jugador.tarjeta_roja = True
        jugador.equipo.tarjetas_rojas += 1
        print(f"Tarjeta roja para {jugador.rol}")
        # Implementar lógica para expulsar al jugador

    def otorgar_tiro_libre(self, jugador_afectado):
        # Implementar lógica para otorgar un tiro libre
        print(f"Tiro libre otorgado a {jugador_afectado.rol}")
        # Aquí puedes agregar la lógica para posicionar el balón y los jugadores para el tiro libre
    # ... (otros métodos del Árbitro) ...
import pygame
import random

class Balon(pygame.sprite.Sprite):
    def __init__(self, x, y, juego):
        super().__init__()
        self.juego = juego
        self.imagen = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(self.imagen, (255, 255, 255), (5, 5), 5)  # White ball
        self.rect = self.imagen.get_rect(center=(x, y))
        self.posicion = pygame.math.Vector2(x, y)
        self.velocidad = pygame.math.Vector2(0, 0)
        self.aceleracion = pygame.math.Vector2(0, 0)
        self.poseedor = None
        self.en_aire = False
        self.altura = 0
        self.friccion = 0.98
        self.gravedad = 0.5

    def actualizar(self):
        if self.poseedor:
            self.seguir_poseedor()
        else:
            self.mover()
            # Comprobar si algún jugador está tocando el balón
            for equipo in [self.juego.equipo1, self.juego.equipo2]:
                for jugador in equipo.jugadores:
                    if self.rect.colliderect(jugador.rect):
                        self.poseedor = jugador
                        break
                if self.poseedor:
                    break
        
        self.aplicar_limites_campo()

    def seguir_poseedor(self):
        self.posicion = pygame.math.Vector2(self.poseedor.rect.center)
        self.rect.center = self.posicion
        self.velocidad = pygame.math.Vector2(0, 0)
        self.altura = 0
        self.en_aire = False

    def mover(self):
        if self.en_aire:
            self.altura -= self.gravedad
            if self.altura <= 0:
                self.altura = 0
                self.en_aire = False
                self.velocidad *= 0.5  # Reduce velocity on bounce

        self.velocidad += self.aceleracion
        self.velocidad *= self.friccion
        self.posicion += self.velocidad
        self.rect.center = self.posicion
        self.aceleracion *= 0

    def aplicar_limites_campo(self):
        if self.rect.left < self.juego.FIELD_START_X:
            self.rect.left = self.juego.FIELD_START_X
            self.velocidad.x *= -0.5
        elif self.rect.right > self.juego.FIELD_START_X + self.juego.FIELD_WIDTH:
            self.rect.right = self.juego.FIELD_START_X + self.juego.FIELD_WIDTH
            self.velocidad.x *= -0.5

        if self.rect.top < self.juego.FIELD_START_Y:
            self.rect.top = self.juego.FIELD_START_Y
            self.velocidad.y *= -0.5
        elif self.rect.bottom > self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT:
            self.rect.bottom = self.juego.FIELD_START_Y + self.juego.FIELD_HEIGHT
            self.velocidad.y *= -0.5

        self.posicion = pygame.math.Vector2(self.rect.center)

    def patear(self, fuerza, direccion):
        self.poseedor = None
        self.velocidad = direccion * fuerza
        self.en_aire = True
        self.altura = random.uniform(10, 30)  # Random initial height

    def esta_en_posesion(self):
        return self.poseedor is not None

    def set_poseedor(self, jugador):
        self.poseedor = jugador
        self.seguir_poseedor()

    def liberar(self):
        self.poseedor = None

    def dibujar(self, pantalla):
        pantalla.blit(self.imagen, self.rect)
        if self.en_aire:
            # Draw shadow
            pygame.draw.ellipse(pantalla, (100, 100, 100), 
                                (self.rect.centerx - 5, self.rect.bottom - 2, 10, 4))

    def reiniciar(self):
        self.posicion = pygame.math.Vector2(self.juego.SCREEN_WIDTH // 2, self.juego.SCREEN_HEIGHT // 2)
        self.rect.center = self.posicion
        self.velocidad = pygame.math.Vector2(0, 0)
        self.aceleracion = pygame.math.Vector2(0, 0)
        self.poseedor = None
        self.en_aire = False
        self.altura = 0
import pygame
import random
from enum import Enum

# Asegúrate de que estas clases estén definidas antes de JuegoFutbol
# class Equipo, class JugadorIA, class Balon, class Arbitro, etc.

class EfectoClima(Enum):
    NORMAL = 1
    LLUVIA = 2
    VIENTO = 3

class JuegoFutbol:
    def __init__(self):
        # Configuración de la pantalla
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Simulación Avanzada de Fútbol IA vs IA")

        # Colores
        self.GREEN = (0, 128, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        # Dimensiones del campo
        self.FIELD_WIDTH = 700
        self.FIELD_HEIGHT = 500
        self.FIELD_START_X = (self.SCREEN_WIDTH - self.FIELD_WIDTH) // 2
        self.FIELD_START_Y = (self.SCREEN_HEIGHT - self.FIELD_HEIGHT) // 2

        # Inicializar equipos
        self.equipo1 = Equipo(self.RED, 'izquierda', self)
        self.equipo2 = Equipo(self.BLUE, 'derecha', self)
        self.equipo1.oponente = self.equipo2
        self.equipo2.oponente = self.equipo1

        # Inicializar balón
        self.balon = Balon(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, self)

        # Variables de estado del juego
        self.marcador = [0, 0]
        self.tiempo = 0
        self.fuente = pygame.font.Font(None, 36)
        self.ultimo_guardado = pygame.time.get_ticks()
        self.numero_partidos = 0
        self.equipo_saque = self.equipo1
        self.escala_tiempo = 20.0
        self.estado_juego = 'saque_inicial'
        self.equipo_saque = random.choice([self.equipo1, self.equipo2])
        self.tiempo_medio = 45 * 60
        self.mitad_actual = 1
        self.retraso_saque = 60  # 1 segundo a 60 FPS


        # Inicializar árbitro
        self.arbitro = Arbitro(self)

        # Variables de clima y ambiente
        self.clima = EfectoClima.NORMAL
        self.direccion_viento = 'derecha'
        self.ultimo_tiempo_actividad = pygame.time.get_ticks()
        self.umbral_inactividad = 10000000  # 10 segundos

        # Estadísticas
        self.estadisticas = {
            'posesion': {self.equipo1: 0, self.equipo2: 0},
            'tiros_a_puerta': {self.equipo1: 0, self.equipo2: 0},
            'pases_completados': {self.equipo1: 0, self.equipo2: 0}
        }

    def actualizar(self):
        self.balon.actualizar()
        self.equipo1.actualizar(self.balon)
        self.equipo2.actualizar(self.balon)

        self.tiempo += 1 / 60 * self.escala_tiempo
        self.actualizar_estadisticas()

        if self.tiempo >= 90 * 60:
            self.finalizar_partido()
            return

        if self.comprobar_fuera_limites():
            return

        self.comprobar_gol()
        self.manejar_tiro_libre()
        self.manejar_corners()
        # Check for excessive ball possession
        if self.balon.poseedor and isinstance(self.balon.poseedor, JugadorIA):
            self.balon.poseedor.actualizar()  # This will trigger the penalty check
        tiempo_actual = pygame.time.get_ticks()
        if tiempo_actual - self.ultimo_guardado > 10000:
            self.guardar_modelos()
            self.ultimo_guardado = tiempo_actual

        if random.random() < 0.00001:
            self.cambiar_clima()

        if self.balon.velocidad.length() < 0.1 and not self.balon.esta_en_posesion() and \
           tiempo_actual - self.ultimo_tiempo_actividad > self.umbral_inactividad:
            self.reiniciar_juego_por_inactividad()
        else:
            self.ultimo_tiempo_actividad = tiempo_actual

        if self.estado_juego == 'saque_inicial':
            if self.retraso_saque > 0:
                self.retraso_saque -= 1
            else:
                self.realizar_saque_inicial()
        elif self.estado_juego == 'en_juego':

            if self.tiempo >= self.tiempo_medio and self.mitad_actual == 1:
                self.evento_medio_tiempo()

        self.arbitro.actualizar()
        self.aplicar_efectos_clima()

    def actualizar_estadisticas(self):
        if self.balon.poseedor:
            self.estadisticas['posesion'][self.balon.poseedor.equipo] += 1

    def comprobar_fuera_limites(self):
        if (self.balon.rect.left <= self.FIELD_START_X or 
            self.balon.rect.right >= self.FIELD_START_X + self.FIELD_WIDTH or
            self.balon.rect.top <= self.FIELD_START_Y or 
            self.balon.rect.bottom >= self.FIELD_START_Y + self.FIELD_HEIGHT):
            self.reiniciar_fuera_limites()
            return True
        return False

    def reiniciar_fuera_limites(self):
        self.balon.rect.center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.balon.posicion = pygame.math.Vector2(self.balon.rect.center)
        self.balon.velocidad = pygame.math.Vector2(0, 0)
        self.balon.poseedor = None
        
        self.equipo1.reiniciar_posiciones()
        self.equipo2.reiniciar_posiciones()
        
        self.estado_juego = 'saque_inicial'
        self.retraso_saque = 60  # 1 segundo a 60 FPS

        self.equipo_saque = random.choice([self.equipo1, self.equipo2])
        
        print("El balón ha salido del campo. Se reinicia el juego desde el centro.")
        self.retraso_saque = 60  # 1 segundo a 60 FPS

    def comprobar_gol(self):
        OFFSET_PORTERIA = -15
        PORTERIA_ALTURA = 150
        if self.equipo1.porteria_propia.colliderect(self.balon.rect):
            self.marcador[1] += 1
            self.penalizar_equipo_por_gol(self.equipo1)
            self.reiniciar_despues_gol(self.equipo2)
            print(f"¡Gol del equipo {self.equipo2.lado}! Marcador: {self.marcador[0]} - {self.marcador[1]}")
        elif self.equipo2.porteria_propia.colliderect(self.balon.rect):
            self.marcador[0] += 1
            self.penalizar_equipo_por_gol(self.equipo2)
            self.reiniciar_despues_gol(self.equipo1)
            print(f"¡Gol del equipo {self.equipo1.lado}! Marcador: {self.marcador[0]} - {self.marcador[1]}")

    def penalizar_equipo_por_gol(self, equipo):
        for jugador in equipo.jugadores:
            if isinstance(jugador, JugadorIA):
                jugador.recibir_penalizacion_por_gol()
        equipo.recibir_penalizacion_por_gol()
    def reiniciar_despues_gol(self, lado):
        self.equipo_saque = self.equipo2 if lado == 'izquierda' else self.equipo1
        self.balon.rect.center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.balon.posicion = pygame.math.Vector2(self.balon.rect.center)
        self.balon.velocidad = pygame.math.Vector2(0, 0)
        self.balon.poseedor = None

        self.equipo1.reiniciar_posiciones()
        self.equipo2.reiniciar_posiciones()

        for jugador in (self.equipo1.jugadores if self.equipo_saque == self.equipo2 else self.equipo2.jugadores):
            if self.equipo_saque == self.equipo2:
                jugador.rect.right = min(jugador.rect.right, self.SCREEN_WIDTH // 2 - 10)
            else:
                jugador.rect.left = max(jugador.rect.left, self.SCREEN_WIDTH // 2 + 10)

        self.estado_juego = 'saque_inicial'
        print(f"¡Gol marcado! El marcador es ahora {self.marcador[0]} - {self.marcador[1]}")
        print(f"Saque inicial para el equipo {self.equipo_saque.lado}")

        self.retraso_saque = 60  # 1 segundo a 60 FPS

    def manejar_tiro_libre(self):
        if self.balon.poseedor and self.balon.poseedor.rect.colliderect(self.balon.rect):
            self.estado_juego = 'en_juego'

    def manejar_corners(self):
        if self.estado_juego == 'corner':
            equipo_atacante = self.equipo1 if self.balon.rect.centerx > self.SCREEN_WIDTH // 2 else self.equipo2
            jugador_corner = min(equipo_atacante.jugadores, key=lambda p: p.distancia_a(self.balon.rect.center))
            
            if jugador_corner.rect.colliderect(self.balon.rect):
                self.balon.poseedor = jugador_corner
                self.estado_juego = 'en_juego'
                print(f"Corner ejecutado por {jugador_corner.rol}")

    def evento_medio_tiempo(self):
        self.mitad_actual = 2
        self.tiempo = 0
        self.equipo1, self.equipo2 = self.equipo2, self.equipo1
        self.equipo1.lado, self.equipo2.lado = 'izquierda', 'derecha'
        self.equipo1.posiciones_base = self.equipo1.calcular_posiciones_base()
        self.equipo2.posiciones_base = self.equipo2.calcular_posiciones_base()
        self.equipo_saque = self.equipo2
        self.estado_juego = 'saque_inicial'
        self.cambiar_clima()

    def finalizar_partido(self):
        print(f"Fin del partido. Resultado final: {self.marcador[0]} - {self.marcador[1]}")
        self.mostrar_estadisticas_finales()
        self.guardar_modelos()
        self.reiniciar_partido()

    def mostrar_estadisticas_finales(self):
        total_posesion = sum(self.estadisticas['posesion'].values())
        posesion_equipo1 = self.estadisticas['posesion'][self.equipo1] / total_posesion * 100
        posesion_equipo2 = self.estadisticas['posesion'][self.equipo2] / total_posesion * 100

        print(f"Estadísticas finales:")
        print(f"Posesión: Equipo 1 {posesion_equipo1:.1f}% - {posesion_equipo2:.1f}% Equipo 2")
        print(f"Tiros a puerta: Equipo 1 {self.estadisticas['tiros_a_puerta'][self.equipo1]} - {self.estadisticas['tiros_a_puerta'][self.equipo2]} Equipo 2")
        print(f"Pases completados: Equipo 1 {self.estadisticas['pases_completados'][self.equipo1]} - {self.estadisticas['pases_completados'][self.equipo2]} Equipo 2")

    def reiniciar_partido(self):
        self.numero_partidos += 1
        self.equipo1, self.equipo2 = self.equipo2, self.equipo1
        self.equipo1.lado, self.equipo2.lado = 'izquierda', 'derecha'
        self.equipo1.reiniciar_equipo()
        self.equipo2.reiniciar_equipo()
        self.marcador = [0, 0]
        self.tiempo = 0
        self.mitad_actual = 1
        self.balon.reiniciar()
        self.estado_juego = 'saque_inicial'
        self.equipo_saque = self.equipo1
        self.estadisticas = {
            'posesion': {self.equipo1: 0, self.equipo2: 0},
            'tiros_a_puerta': {self.equipo1: 0, self.equipo2: 0},
            'pases_completados': {self.equipo1: 0, self.equipo2: 0}
        }
        print(f"Nuevo partido iniciado. Partido número: {self.numero_partidos}")

    def cambiar_clima(self):
        self.clima = random.choice(list(EfectoClima))
        if self.clima == EfectoClima.VIENTO:
            self.direccion_viento = random.choice(['izquierda', 'derecha'])
        print(f"El clima ha cambiado a: {self.clima.name}")

    def aplicar_efectos_clima(self):
        if self.clima == EfectoClima.LLUVIA:
            for equipo in [self.equipo1, self.equipo2]:
                for jugador in equipo.jugadores:
                    jugador.velocidad_maxima *= 0.9
                    if random.random() < 0.01:
                        jugador.velocidad_vector *= 0.5
        elif self.clima == EfectoClima.VIENTO:
            if not self.balon.poseedor:
                viento = pygame.math.Vector2(5 if self.direccion_viento == 'derecha' else -5, 0)
                self.balon.velocidad += viento * 0.1


    def reiniciar_juego_por_inactividad(self):
        print("Juego reiniciado por inactividad")
        self.balon.rect.center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.balon.posicion = pygame.math.Vector2(self.balon.rect.center)
        self.balon.velocidad = pygame.math.Vector2(0, 0)
        self.balon.poseedor = None
        self.estado_juego = 'saque_inicial'
        self.equipo_saque = random.choice([self.equipo1, self.equipo2])
        self.ultimo_tiempo_actividad = pygame.time.get_ticks()

        self.equipo1.reiniciar_posiciones()
        self.equipo2.reiniciar_posiciones()

        for equipo in [self.equipo1, self.equipo2]:
            for jugador in equipo.jugadores:
                jugador.puntuacion -= 20
                jugador.estado = EstadoJugador.INACTIVO

        print(f"Todos los jugadores penalizados y reposicionados. Saque inicial para el equipo {self.equipo_saque.lado}")
        self.retraso_saque = 60  # 1 segundo a 60 FPS

    def realizar_saque_inicial(self):
        print("Realizando saque inicial")
        self.equipo1.reiniciar_posiciones()
        self.equipo2.reiniciar_posiciones()

        self.balon.rect.center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.balon.posicion = pygame.math.Vector2(self.balon.rect.center)
        self.balon.velocidad = pygame.math.Vector2(0, 0)
        self.balon.poseedor = None

        jugador_saque = min(self.equipo_saque.jugadores, key=lambda p: p.distancia_a(self.balon.rect.center))
        offset = -10 if self.equipo_saque.lado == 'izquierda' else 10  # Reducimos el offset
        jugador_saque.rect.center = (self.balon.rect.centerx + offset, self.balon.rect.centery)

        for jugador in (self.equipo1.jugadores if self.equipo_saque == self.equipo2 else self.equipo2.jugadores):
            if self.equipo_saque == self.equipo2:
                jugador.rect.right = min(jugador.rect.right, self.SCREEN_WIDTH // 2 - 10)
            else:
                jugador.rect.left = max(jugador.rect.left, self.SCREEN_WIDTH // 2 + 10)

        self.estado_juego = 'en_juego'
        print("Saque inicial realizado")

    def reiniciar_despues_gol(self, lado):
        # ... (código existente) ...
        self.estado_juego = 'saque_inicial'
        self.equipo_saque = self.equipo2 if lado == 'izquierda' else self.equipo1
        self.retraso_saque = 60  # 1 segundo a 60 FPS
    def guardar_modelos(self):
        for equipo in [self.equipo1, self.equipo2]:
            for jugador in equipo.jugadores:
                if isinstance(jugador, JugadorIA):
                    ruta_modelo = f"modelo_{jugador.equipo.lado}_{jugador.rol}.pth"
                    jugador.guardar_modelo(ruta_modelo)
        print("Modelos de jugadores guardados")

    def dibujar(self, pantalla):
        pantalla.fill(self.GREEN)
        pygame.draw.rect(pantalla, self.WHITE, (self.FIELD_START_X, self.FIELD_START_Y, self.FIELD_WIDTH, self.FIELD_HEIGHT), 2)
        pygame.draw.circle(pantalla, self.WHITE, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), 50, 2)

        self.dibujar_porterias(pantalla)
        self.equipo1.dibujar(pantalla)
        self.equipo2.dibujar(pantalla)
        self.balon.dibujar(pantalla)

        self.dibujar_interfaz(pantalla)

    def dibujar_porterias(self, pantalla):
        OFFSET_PORTERIA = 20
        PORTERIA_ALTURA = 150
        PORTERIA_ANCHO = 10

        pygame.draw.rect(pantalla, self.WHITE, (self.FIELD_START_X - PORTERIA_ANCHO - OFFSET_PORTERIA, 
                                           self.FIELD_START_Y + self.FIELD_HEIGHT // 2 - PORTERIA_ALTURA // 2, 
                                           PORTERIA_ANCHO, PORTERIA_ALTURA), 2)
        pygame.draw.rect(pantalla, self.WHITE, (self.FIELD_START_X + self.FIELD_WIDTH + OFFSET_PORTERIA, 
                                           self.FIELD_START_Y + self.FIELD_HEIGHT // 2 - PORTERIA_ALTURA // 2, 
                                           PORTERIA_ANCHO, PORTERIA_ALTURA), 2)

    def dibujar_interfaz(self, pantalla):
        texto_marcador = self.fuente.render(f"{self.marcador[0]} - {self.marcador[1]}", True, self.WHITE)
        texto_tiempo = self.fuente.render(f"Tiempo: {int(self.tiempo // 60)}:{int(self.tiempo % 60):02d}", True, self.WHITE)
        texto_mitad = self.fuente.render(f"Mitad: {self.mitad_actual}", True, self.WHITE)
        texto_clima = self.fuente.render(f"Clima: {self.clima.name}", True, self.WHITE)

        pantalla.blit(texto_marcador, (self.SCREEN_WIDTH // 2 - texto_marcador.get_width() // 2, 10))
        pantalla.blit(texto_tiempo, (10, 10))
        pantalla.blit(texto_mitad, (self.SCREEN_WIDTH - texto_mitad.get_width() - 10, 10))
        pantalla.blit(texto_clima, (10, 40))

        if self.clima == EfectoClima.VIENTO:
            texto_viento = self.fuente.render(f"Viento: {'→' if self.direccion_viento == 'derecha' else '←'}", True, self.WHITE)
            pantalla.blit(texto_viento, (10, 70))

        texto_faltas = self.fuente.render(f"Faltas: {self.arbitro.conteo_faltas}", True, self.WHITE)
        pantalla.blit(texto_faltas, (self.SCREEN_WIDTH - texto_faltas.get_width() - 10, 40))

    def ejecutar(self):
        reloj = pygame.time.Clock()
        ejecutando = True

        while ejecutando:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    ejecutando = False

            self.actualizar()
            self.dibujar(self.screen)
            pygame.display.flip()
            reloj.tick(60)  # Mantener 60 FPS

        pygame.quit()

# Función principal
def main():
    pygame.init()
    juego = JuegoFutbol()
    juego.ejecutar()

if __name__ == "__main__":
    main()