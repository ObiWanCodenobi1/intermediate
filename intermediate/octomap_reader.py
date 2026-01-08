import numpy as np
import pyoctomap

tree = pyoctomap.OcTree(0.15)
#file = open("icuas26_1.bt","rb")
map = tree.readBinary("icuas26_1.bt")
#file.close()

max_x = max_y = max_z = min_x = min_y = min_z = count = 0

for leaf in tree.begin_leafs():
    x,y,z = tuple(leaf.getCoordinate())
    count += 1
    if x> max_x: max_x = x
    if y> max_y: max_y = y
    if z> max_z: max_z = z
    if x< min_x: min_x = x
    if y< min_y: min_y = y
    if z< min_z: min_z = z

print(max_x,max_y,max_z)
print(min_x,min_y,min_z)
print(count)

