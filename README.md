# insibi
The **in**-**si**lico **bi**ome is an artificial life simulation using physical particle cells.
Creatures use recurrent neural networks as protein system and have the ability to evolve into 
multicellular plants and animals.

![example image](https://user-images.githubusercontent.com/22052799/29769289-0b09dd40-8bea-11e7-9941-d4ed750ed534.png)

## Dependencies
Insibi is dependent on *numpy*, *pymunk*, *pygame*, *cv2* and *scipy*.

## Quickstart
When running insibi, it enters simulation mode. By entering "d" or "display" it goes into visualization mode.
In Vismode, you can move around the world using arrow keys, zoom in/out using *,* and *.*, left-click a cell to see its stats and right-click to feed them.
Access the following functions by hitting keys:
* **l** display light
* **s** display solvent
  * red: enzyme
  * green: waste
  * blue: nutrient
* **o** display odor (red, green and blue are different odors)
* **x** display commitment of cell to livestyle
  * red: enzyme expression system
  * green: chlorophyll level
  * blue: transport protein level
* **r** reseed with random cells
* **f** fullscreen mode on/off
* **space** stop/start simulation

## Comments
It takes a while for the cells to develop complex behavior. One night in minimal mode is usually enough to observe speciation and differentiation.
