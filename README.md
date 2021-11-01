# Self Organization Maps Network Architecture?

Pour les besoins, nous discuterons d'un SOM à deux dimensions. Le réseau est créé à partir d'un réseau 2D de « nœuds », dont chacun est entièrement connecté à la couche d'entrée. La figure ci-dessous montre un très petit réseau Kohonen de 4 X 4 nœuds connectés à la couche d'entrée (en vert) représentant un vecteur bidimensionnel.

![image 1](images/1.png)

Chaque nœud a une position topologique spécifique (une coordonnée x, y dans le réseau) et contient un vecteur de poids de même dimension que les vecteurs d'entrée. C'est-à-dire, si les données d'apprentissage sont constituées de vecteurs, V, de n dimensions :<br/>
<code>V1, V2, V3…Vn</code>

Ensuite, chaque nœud contiendra un vecteur de poids correspondant W, de n dimensions :<br/>

<code>W1, W2, W3…Wn</code>

## Comment fonctionnent les cartes d'auto-organisation ?

### Présentation de l'algorithme d'apprentissage

Un SOM n'a pas besoin de spécifier une sortie cible contrairement à de nombreux autres types de réseau. Au lieu de cela, lorsque les poids des nœuds correspondent au vecteur d'entrée, cette zone du réseau est sélectivement optimisée pour ressembler plus étroitement aux données de la classe dont le vecteur d'entrée est membre. A partir d'une distribution initiale de poids aléatoires, et sur de nombreuses itérations, le SOM finit par s'installer dans une carte de zones stables. Chaque zone est en fait un classificateur d'entités, vous pouvez donc considérer la sortie graphique comme un type de carte d'entités de l'espace d'entrée.

L'entrainement se déroule en plusieurs étapes et en plusieurs itérations :

1. Les poids de chaque nœud sont initialisés.
   
2. Un vecteur est choisi au hasard dans l'ensemble des données d'apprentissage et présenté au réseau.
   
3. Chaque nœud est examiné pour calculer les poids qui ressemblent le plus au vecteur d'entrée. Le nœud gagnant est communément appelé Best Matching Unit (BMU).
   
4. Le rayon du voisinage de la BMU est maintenant calculé. Il s'agit d'une valeur qui commence en grand, généralement définie sur le « rayon » du réseau, mais diminue à chaque pas de temps. Tous les nœuds trouvés dans ce rayon sont réputés être à l'intérieur du voisinage de la BMU.
   
5. Les poids de chaque nœud voisin (les nœuds trouvés à l'étape 4) sont ajustés pour les rendre plus semblables au vecteur d'entrée. Plus un nœud est proche de la BMU ; plus ses poids s'altèrent.
   
6. Répétez l'étape 2 pour N itérations.