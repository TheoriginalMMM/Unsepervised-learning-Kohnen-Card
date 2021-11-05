# coding: utf8
#!/usr/bin/env python
# ------------------------------------------------------------------------
# Carte de Kohonen
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------
# Implémentation de l'algorithme des cartes auto-organisatrices de Kohonen
# ------------------------------------------------------------------------
# Pour que les divisions soient toutes réelles (pas de division entière)
from __future__ import division
# Librairie de calcul matriciel
import numpy
from math import exp
import time

# Librairie d'affichage
import matplotlib.pyplot as plt

class Neuron:
  ''' Classe représentant un neurone '''
  
  def __init__(self, w, posx, posy):
    '''
    @summary: Création d'un neurone
    @param w: poids du neurone
    @type w: numpy array
    @param posx: position en x du neurone dans la carte
    @type posx: int
    @param posy: position en y du neurone dans la carte
    @type posy: int
    '''
    # Initialisation des poids
    self.weights = w.flatten()
    # Initialisation de la position
    self.posx = posx
    self.posy = posy
    # Initialisation de la sortie du neurone
    self.y = 0.
  
  def compute(self,x):
    '''
    @summary: Affecte à y la valeur de sortie du neurone (i.e. la distance entre son poids et l'entrée)
    @param x: entrée du neurone
    @type x: numpy array
    '''
    # TODO
    self.y = numpy.linalg.norm(x-self.weights)

  
  def computeForPrediction(self,x,inverse=False):
    '''
    @summary: Affecte à y distance la distance entre une partie de son poids et l'entrée(utiliser pour la prédiction)
    @param x: entrée (cordonné spatiale ou motrice)
    @type x: numpy array
    '''
    if(not inverse):
      self.y = numpy.linalg.norm(x-self.weights[:2])
    else: 
      self.y = numpy.linalg.norm(x-self.weights[2:])
  
  def learn(self,eta,sigma,posxbmu,posybmu,x):
    '''
    @summary: Modifie les poids selon la règle de Kohonen
    @param eta: taux d'apprentissage
    @type eta: float
    @param sigma: largeur du voisinage
    @type sigma: float
    @param posxbmu: position en x du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
    @type posxbmu: int
    @param posybmu: position en y du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
    @type posybmu: int
    @param x: entrée du neurone
    @type x: numpy array
    '''
    # TODO (attention à ne pas changer la partie à gauche du =)
    ME = numpy.array((self.posx,self.posy))
    BMU = numpy.array((posxbmu, posybmu))
    Distance = (numpy.linalg.norm(ME-BMU))**2
    dif = x-self.weights
    distcaree=(-Distance)/(2*(sigma**2))
    self.weights[:] = self.weights + (eta*dif*exp(distcaree))


class SOM:
  ''' Classe implémentant une carte de Kohonen. '''

  def __init__(self, inputsize, gridsize):
    '''
    @summary: Création du réseau
    @param inputsize: taille de l'entrée
    @type inputsize: tuple
    @param gridsize: taille de la carte
    @type gridsize: tuple
    '''
    # Initialisation de la taille de l'entrée
    self.inputsize = inputsize
    # Initialisation de la taille de la carte
    self.gridsize = gridsize
    # Création de la carte
    # Carte de neurones
    self.map = []    
    # Carte des poids
    self.weightsmap = []
    # Carte des activités
    self.activitymap = []
    for posx in range(gridsize[0]):
      mline = []
      wmline = []
      amline = []
      for posy in range(gridsize[1]):
        neuron = Neuron(numpy.random.random(self.inputsize),posx,posy)
        mline.append(neuron)
        wmline.append(neuron.weights)
        amline.append(neuron.y)
      self.map.append(mline)
      self.weightsmap.append(wmline)
      self.activitymap.append(amline)
    self.activitymap = numpy.array(self.activitymap)

  def compute(self,x):
    '''
    @summary: calcule de l'activité des neurones de la carte
    @param x: entrée de la carte (identique pour chaque neurone)
    @type x: numpy array
    '''
    # On demande à chaque neurone de calculer son activité et on met à jour la carte d'activité de la carte
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].compute(x)
        self.activitymap[posx][posy] = self.map[posx][posy].y

  def computeForPredection(self,x,inverse=False):
    '''
    @summary: calcule la distance entre une cordonné et la bonne partie des neurones de la carte
    @param x: cordonné en entrée qu'on veut estimer sa deuxiemme partie
    @type x: numpy array
    '''
    # On demande à chaque neurone de calculer son activité et on met à jour la carte d'activité de la carte
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].computeForPrediction(x,inverse)
        self.activitymap[posx][posy] = self.map[posx][posy].y

  def learn(self,eta,sigma,x):
    '''
    @summary: Modifie les poids de la carte selon la règle de Kohonen
    @param eta: taux d'apprentissage
    @type eta: float
    @param sigma: largeur du voisinage
    @type sigma: float
    @param x: entrée de la carte
    @type x: numpy array
    '''
    # Calcul du neurone vainqueur
    bmux,bmuy = numpy.unravel_index(numpy.argmin(self.activitymap),self.gridsize)
    # Mise à jour des poids de chaque neurone
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].learn(eta,sigma,bmux,bmuy,x)

  def scatter_plot(self,interactive=False):
    '''
    @summary: Affichage du réseau dans l'espace d'entrée (utilisable dans le cas d'entrée à deux dimensions et d'une carte avec une topologie de grille carrée)
    @param interactive: Indique si l'affichage se fait en mode interactif
    @type interactive: boolean
    '''
    # Création de la figure
    if not interactive:
      plt.figure()
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
    # Modification des limites de l'affichage
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace d\'entree')
    # Affichage de la figure
    if not interactive:
      plt.show()

  def scatter_plot_2(self,interactive=False):
    '''
    @summary: Affichage du réseau dans l'espace d'entrée en 2 fois 2d (utilisable dans le cas d'entrée à quatre dimensions et d'une carte avec une topologie de grille carrée)
    @param interactive: Indique si l'affichage se fait en mode interactif
    @type interactive: boolean
    '''
    # Création de la figure
    if not interactive:
      plt.figure()
    # Affichage des 2 premières dimensions dans le plan
    plt.subplot(1,2,1)
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
    # Affichage des 2 dernières dimensions dans le plan
    plt.subplot(1,2,2)
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,2].flatten(),w[:,:,3].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,2],w[i,:,3],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,2],w[:,i,3],'k',linewidth=1.)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace d\'entree')
    # Affichage de la figure
    if not interactive:
      plt.show()

  def plot(self):
    '''
    @summary: Affichage des poids du réseau (matrice des poids)
    '''
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Création de la figure
    f,a = plt.subplots(w.shape[0],w.shape[1])    
    # Affichage des poids dans un sous graphique (suivant sa position de la SOM)
    for i in range(w.shape[0]):
      for j in range(w.shape[1]):
        plt.subplot(w.shape[0],w.shape[1],i*w.shape[1]+j+1)
        im = plt.imshow(w[i,j].reshape(self.inputsize),interpolation='nearest',vmin=numpy.min(w),vmax=numpy.max(w),cmap='binary')
        plt.xticks([])
        plt.yticks([])
    # Affichage de l'échelle
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace de la carte')
    # Affichage de la figure
    plt.show()

  def MSE(self,X):
    '''
    @summary: Calcul de l'erreur de quantification vectorielle moyenne du réseau sur le jeu de données
    @param X: le jeu de données
    @type X: numpy array
    '''
    # On récupère le nombre d'exemples
    nsamples = X.shape[0]
    # Somme des erreurs quadratiques
    s = 0
    # Pour tous les exemples du jeu de test
    for x in X:
      # On calcule la distance à chaque poids de neurone
      self.compute(x.flatten())
      # On rajoute la distance minimale au carré à la somme
      s += numpy.min(self.activitymap)**2
    # On renvoie l'erreur de quantification vectorielle moyenne
    return s/nsamples

  def MAO(self):
    '''
    @summary: Calcul la plus grande distance entre tous  les poids des neurones 
    return: la plus grande distance entre deux neurones de la carte aprés auto-organisation
    @type: float
    '''
    bestdis = 0.
    iterDIS =0.
    for i in range(self.gridsize[0]):
      for j in range(self.gridsize[1]):
        MW = self.weightsmap[i][j]
        for k in range(self.gridsize[0]):
          for l in range(self.gridsize[1]):
            iterDIS=numpy.linalg.norm(MW-self.weightsmap[k][l])
            if(bestdis<iterDIS):
              bestdis=iterDIS

    print("Mésure d\'auto organisation MAO : ",bestdis)

  def SpatialToMotriceORInversseMoyenne(self,spatial,inverse=False):
    '''
    @summary: Passer d'une cordonés spatial à une  cordonées motrice ou l'inveresse en utilisant la moyenne cfrapport
    @param spatial: la cordoné en entré qu'on veut trouvé sa deuximme partie
    @type spatial: numpy array de taille deux 
    @param inverse : Paramatre pour utiliser l'inverse de notre fonction (ie:pour passer d'une cordonné motrice à une cordoné spatial faut le mettre à vrai)
    @type inverse : Boolean 
    '''
    self.computeForPredection(spatial,inverse)
    bmux,bmuy = numpy.unravel_index(numpy.argmin(self.activitymap),self.gridsize)
    DistanceCumule=0
    if( not inverse):
      DistanceBest=numpy.linalg.norm(spatial-self.weightsmap[bmux][bmuy][:2])
      if(DistanceBest==0):
        return self.weightsmap[bmux][bmuy][2:]
      DistanceCumule+=1/DistanceBest
      RESULTAT=self.weightsmap[bmux][bmuy][2:]*1/DistanceBest
      if (spatial[0]-self.weightsmap[bmux][bmuy][0]<=0): 
        distanceAdroite=numpy.linalg.norm(spatial-self.weightsmap[(bmux+1)%self.gridsize[0]][bmuy][:2])
        RESULTAT+=self.weightsmap[(bmux+1)%self.gridsize[0]][bmuy][2:]*1/distanceAdroite
        DistanceCumule+=1/distanceAdroite
        if(spatial[1]-self.weightsmap[bmux][bmuy][1]<=0):
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(bmuy+1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[bmux][(bmuy+1)%self.gridsize[1]][2:]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(bmux+1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[(bmux+1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][2:]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT
        else:
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(bmux+1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[(bmux+1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT
      else:
        distanceAdroite=numpy.linalg.norm(spatial-self.weightsmap[(self.gridsize[1]+bmux-1)%self.gridsize[0]][bmuy][:2])
        RESULTAT+=self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][bmuy][2:]*1/distanceAdroite
        DistanceCumule+=1/distanceAdroite

        if(spatial[1]-self.weightsmap[bmux][bmuy][1]<=0):
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(bmuy+1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[bmux][(bmuy+1)%self.gridsize[0]][2:]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(self.gridsize[1]+bmux-1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][2:]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT

        else:
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2])
          RESULTAT+=self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT
      
      return RESULTAT/DistanceCumule
    else:
      DistanceBest=numpy.linalg.norm(spatial-self.weightsmap[bmux][bmuy][2:])
      if(DistanceBest==0):
        return self.weightsmap[bmux][bmuy][:2]
      DistanceCumule+=1/DistanceBest
      RESULTAT=self.weightsmap[bmux][bmuy][2:]*1/DistanceBest
      if (spatial[0]-self.weightsmap[bmux][bmuy][0]<=0): 
        distanceAdroite=numpy.linalg.norm(spatial-self.weightsmap[(bmux+1)%self.gridsize[0]][bmuy][2:])
        RESULTAT+=self.weightsmap[(bmux+1)%self.gridsize[0]][bmuy][:2]*1/distanceAdroite
        DistanceCumule+=1/distanceAdroite
        if(spatial[1]-self.weightsmap[bmux][bmuy][1]<=0):
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(bmuy+1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[bmux][(bmuy+1)%self.gridsize[1]][:2]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(bmux+1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[(bmux+1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][:2]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT
        else:
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(bmux+1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[(bmux+1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT
      else:
        distanceAdroite=numpy.linalg.norm(spatial-self.weightsmap[(self.gridsize[1]+bmux-1)%self.gridsize[0]][bmuy][2:])
        RESULTAT+=self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][bmuy][:2]*1/distanceAdroite
        DistanceCumule+=1/distanceAdroite

        if(spatial[1]-self.weightsmap[bmux][bmuy][1]<=0):
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(bmuy+1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[bmux][(bmuy+1)%self.gridsize[0]][:2]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(self.gridsize[1]+bmux-1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][(bmuy+1)%self.gridsize[1]][:2]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT

        else:
          distanceHaut=numpy.linalg.norm(spatial-self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[bmux][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2]*1/distanceHaut
          DistanceCumule+=1/distanceHaut
          DistanceHAUTDROIT=numpy.linalg.norm(spatial-self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][2:])
          RESULTAT+=self.weightsmap[(self.gridsize[0]+bmux-1)%self.gridsize[0]][(self.gridsize[1]+bmuy-1)%self.gridsize[1]][:2]*1/DistanceHAUTDROIT
          DistanceCumule+=1/DistanceHAUTDROIT
      return RESULTAT/DistanceCumule

  def SpatialToMotriceORInversse(self,spatial,inverse=False):
    '''
    @summary: Passer des cordonés spatial aux cordonées motrices ou l'inversse 
    @param spatial: les cordonnés spacial
    @type X: numpy array de taille deux 
    @param inverse : Paramatre pour utiliser l'inverse de notre fonction (ie:pour passer d'une cordonné motrice à une cordoné spatial faut le mettre à vrai)
    @type inverse : Boolean 
    '''
    self.computeForPredection(spatial,inverse)
    bmux,bmuy = numpy.unravel_index(numpy.argmin(self.activitymap),self.gridsize)
    if( not inverse):
      return self.weightsmap[bmux][bmuy][2:]
    else:
      return self.weightsmap[bmux][bmuy][:2]
  def EstimerTrajectoireSpatial(self,Depart,Arrive,Steps,Moyenne=False):
    '''
    @summary: Estimer la trajectoire spatial du bras du robot d'un point motrice D à un point D avec deux méthode differentes
    @param Arriver: cordonné motrice d'arrivé 
    @type Arriver: numpy array de taille deux 
    @param Depart: cordonné motrice de départ 
    @type X: numpy array de taille deux 
    @param Steps : nombre de point a tracer pour la trajéctoire
    @type Steps : int 
    @param Moyenne : Paramatre pour utiliser l'inverse de notre fonction (ie:pour passer d'une cordonné motrice à une cordoné spatial faut le mettre à vrai)
    @type inverse : Boolean 
    '''
    diff1=(Arrive[0]-Depart[0])/Steps
    diff2=(Arrive[1]-Depart[1])/Steps
    Res=numpy.array
    for i in range(Steps+1):
      if(i==0):
        if(not Moyenne):
          Res=numpy.array(self.SpatialToMotriceORInversse(numpy.array([Depart[0]+((i)*diff1),Depart[1]+(i*diff2)]),True)).reshape(1,2)
        else:
          s=(Steps+1,2)
          Res= numpy.zeros(s)
          Res[i,:]=numpy.array(self.SpatialToMotriceORInversseMoyenne(numpy.array([Depart[0]+(i*diff1),Depart[1]+(i*diff2)]),True)).reshape(1,2)
      else:
        if(not Moyenne):
          NouvelleCordonne=(self.SpatialToMotriceORInversse(numpy.array([Depart[0]+(i*diff1),Depart[1]+(i*diff2)]),True).reshape(1,2))
          if(not ((numpy.equal(Res[Res.shape[0]-1,:],NouvelleCordonne).all()))):
            Res=numpy.append(Res,NouvelleCordonne,axis=0)

        else:
          NouvelleCordonne=(self.SpatialToMotriceORInversseMoyenne(numpy.array([Depart[0]+(i*diff1),Depart[1]+(i*diff2)]),True).reshape(1,2))
          Res[i,:]=NouvelleCordonne
    
    if(Moyenne):
      print("La trajectoire prédit en utilisant la moyenne :")
    else:
      print("La trajectoire prédit sans utiliser  la moyenne :")
    print(Res)
    return Res
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  # Création d'un réseau avec une entrée (2,1) et une carte (10,10)
  #TODO mettre à jour la taille des données d'entrée pour les données robotiques
  network = SOM((4,1),(10,10))
  # PARAMÈTRES DU RÉSEAU
  # Taux d'apprentissage
  ETA = 0.09 
  # Largeur du voisinage
  SIGMA = 1.4
  # Nombre de pas de temps d'apprentissage
  N =30000
  # Affichage interactif de l'évolution du réseau 
  #TODO à mettre à faux pour que les simulations aillent plus vite
  VERBOSE = False
  # Nombre de pas de temps avant rafraissichement de l'affichage
  NAFFICHAGE = 1000
  # DONNÉES D'APPRENTISSAGE
  # Nombre de données à générer pour les ensembles 1, 2 et 3
  # TODO décommenter les données souhaitées
  nsamples = 1200

  # Ensemble de données 1
  #samples = numpy.random.random((nsamples,2,1))*2-1
  #Ensemble de données 2
  #samples1 = -numpy.random.random((nsamples//3,2,1))
  #samples2 = numpy.random.random((nsamples//3,2,1))
  #samples2[:,0,:] -= 1
  #samples3 = numpy.random.random((nsamples//3,2,1))
  #samples3[:,1,:] -= 1
  #samples = numpy.concatenate((samples1,samples2,samples3))
  # Ensemble de données 3
  #samples1 = numpy.random.random((nsamples//2,2,1))
  #samples1[:,0,:] -= 1
  #samples2 = numpy.random.random((nsamples//2,2,1))
  #samples2[:,1,:] -= 1
  #samples = numpy.concatenate((samples1,samples2))
#Notre ensembe de donné personalisé (non uniforme )

  #samples1 = numpy.random.random((5*nsamples//6,2,1))
  #samples2 = -numpy.random.random((nsamples//6,2,1))
  #samples = numpy.concatenate((samples1,samples2))
  #samples = numpy.concatenate((samples1,samples1,samples1,samples1,samples1,samples2))
# Ensemble de données robotiques
  samples = numpy.random.random((nsamples,4,1))
  samples[:,0:2,:] *= numpy.pi
  l1 = 0.7
  l2 = 0.3
  samples[:,2,:] = l1*numpy.cos(samples[:,0,:])+l2*numpy.cos(samples[:,0,:]+samples[:,1,:])
  samples[:,3,:] = l1*numpy.sin(samples[:,0,:])+l2*numpy.sin(samples[:,0,:]+samples[:,1,:])
  # Affichage des données (pour les ensembles 1, 2 et 3)
  #plt.figure()
  #plt.scatter(samples[:,0,0], samples[:,1,0])
  #plt.xlim(-1,1)
  #plt.ylim(-1,1)
  #plt.suptitle('Donnees apprentissage')
  #plt.show()
  # Affichage des données (pour l'ensemble robotique)
  #plt.figure()
  #plt.subplot(1,2,1)
  #plt.scatter(samples[:,0,0].flatten(),samples[:,1,0].flatten(),c='k')
  #plt.subplot(1,2,2)
  #plt.scatter(samples[:,2,0].flatten(),samples[:,3,0].flatten(),c='k')
  #plt.suptitle('Donnees apprentissage')
  #plt.show()
  # SIMULATION
  # Affichage des poids du réseau
  #network.plot()
  # Initialisation de l'affichage interactif
  if VERBOSE:
    # Création d'une figure
    plt.figure()
    # Mode interactif
    plt.ion()
    # Affichage de la figure
    plt.show()
  start_time = time.time()
  # Boucle d'apprentissage
  for i in range(N+1):
    # Choix d'un exemple aléatoire pour l'entrée courante
    index = numpy.random.randint(nsamples)
    x = samples[index].flatten()
    # Calcul de l'activité du réseau
    network.compute(x)
    # Modification des poids du réseau
    network.learn(ETA,SIGMA,x)
    # Mise à jour de l'affichage
    if VERBOSE and i%NAFFICHAGE==0:
      # Effacement du contenu de la figure
      plt.clf()
      # Remplissage de la figure
      # TODO à remplacer par scatter_plot_2 pour les données robotiques
      #network.scatter_plot_2(True)
      network.scatter_plot(True)
      # Affichage du contenu de la figure
      plt.pause(0.00001)
      plt.draw()
  # Fin de l'affichage interactif
  #network.scatter_plot()
  #network.scatter_plot_2()
  if VERBOSE:
    # Désactivation du mode interactif
    plt.ioff()
  print("---  Temps d'apprentisage de la carte en secondes  %s seconds ---" % (time.time() - start_time))
  # Affichage des poids du réseau
  #network.plot()
  # Affichage de l'erreur de quantification vectorielle moyenne après apprentissage
  print("TAUX D'APPRENTISAGE ",ETA)
  print("Longeur de voisingage ",SIGMA)
  print("Nombre de pas d'apprentissage ",N)
  print("erreur de quantification vectorielle moyenne ",network.MSE(samples))
  network.MAO()
  #condition a mettre a faux si on veut pas tester et voir les perfermonces des méthdoes d'estimations de cordonnée
  if(True):
    ERREURS=numpy.zeros((4))
    for i in range(nsamples):
    # Choix d'un exemple aléatoire pour l'entrée courante
      #index = numpy.random.randint(nsamples)
      cordS=samples[i,:2]
      cordM=samples[i,2:]
      spatialEM=network.SpatialToMotriceORInversseMoyenne(cordM,True)
      spatialE=network.SpatialToMotriceORInversse(cordM,True)
      motriceEM=network.SpatialToMotriceORInversseMoyenne(cordS)
      motriceE=network.SpatialToMotriceORInversse(cordS)
      ERREURS+=numpy.array([numpy.linalg.norm(cordS-spatialE),numpy.linalg.norm(cordS-spatialEM),numpy.linalg.norm(cordM-motriceE),numpy.linalg.norm(cordM-motriceEM)])

    ERREURS=ERREURS/nsamples
    print("Erreur moyenne pour la méthode MotriceToSpatial SANS Moyenne est :",ERREURS[0])
    print("Erreur moyenne pour la méthode MotriceToSpatial Avec Moyenne est :",ERREURS[1])
    print("Erreur moyenne pour la méthode SpatialToMotrice Sans Moyenne est :",ERREURS[2])
    print("Erreur moyenne pour la méthode SpatialToMotrice Avec Moyenne est :",ERREURS[3])
    
