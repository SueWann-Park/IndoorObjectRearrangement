import numpy as np
import open3d as o3d
import os
import math
from os.path import join
from os.path import isfile
from os.path import isdir
import random
import copy

def rearrange():
    files = os.listdir(join('fetched'))
    empty = [f for f in files if isfile(join('fetched', f)) == True and 'empty' in f][0]
    width, height = int(empty.split('_')[1]), int(empty.split('_')[2])
    desk_list, pcd_addrs = unpack_data()
    print(desk_list)
    
    room = np.zeros([width, height])
    print("Room Shape: " + str(room.shape))
    #pcds = [o3d.io.read_point_cloud(join('fetched', empty))]
    
    box = o3d.io.read_triangle_mesh(join('fetched', 'box.ply'))
    line = o3d.io.read_line_set(join('fetched', 'line.ply'))
    #mesh1.paint_uniform_color([1, 0.706, 0])
    pcds = [box, line]
    #o3d.visualization.draw_geometries(pcds)
    
    success, result = rec_place_objects(room, pcd_addrs, pcds, 10, 1, desk_list)
    if success == True:
        _room_, _pcd_addrs, _pcds = result
        print(_pcds)
        print_room(_room_)
        
        pcd_combined = o3d.geometry.PointCloud()
        for point_id in range(2, len(_pcds)):
            pcd_combined += _pcds[point_id]
        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.05)
        o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined)
        o3d.visualization.draw_geometries([_pcds[0], _pcds[1], pcd_combined_down])
        #o3d.visualization.draw_geometries(_pcds)

def rec_place_objects(_room, _pcd_addrs, _pcds, numTry, numObject, desk_list):
    
    if len(_pcd_addrs) == 0:
        return True, (_room, _pcd_addrs, _pcds)
    if numTry == 0:
        return False, None
    
    room = copy.deepcopy(_room)
    pcd_addrs = copy.deepcopy(_pcd_addrs)
    pcds = copy.deepcopy(_pcds)
    
    pcd_addr = pcd_addrs[0]
    pcd_addrs.remove(pcd_addr)
    
    pcd = o3d.io.read_point_cloud(pcd_addr)
    w,h = int(pcd_addr.split('_')[1]), int(pcd_addr.split('_')[2])
    print("Object Shape: \t" + str(w) + ",\t" + str(h))
    print(pcd_addr)
    
    if 'O_' in pcd_addr:
        result = available_boundary(room, room.shape[0], room.shape[1])
        boundary = [] # L, R, T, B
        for i in range(0, len(result)):            
            L_max, L_index = result[i]
            if L_max >= h:
                boundary.append(i)
        if len(boundary) == 0:
            return False, None
        
        if len(boundary) == 0:
            return False, None
        
        edge = random.choice(boundary)
        print("Boundary: ", boundary)
        
        x_max, y_max = result[edge][1]
        L_max = result[edge][0]
        print("\tL_max: " + str(L_max))
        print("\tx_max: " + str(x_max) + ", y_max: " + str(y_max))
        
        if(edge == 0):   #L
            offset = y_max - L_max + random.choice([k for k in range(0, y_max - (y_max - L_max + h))])
            pcd.translate(np.array([(w/2) / 100.0, (h/2) / 100.0, 0]), relative = True)
            pcd.translate(np.array([0, offset / 100.0, 0]), relative = True)
        elif(edge == 1): #R
            offset = y_max - L_max + random.choice([k for k in range(0, y_max - (y_max - L_max + h))])
            pcd = rotatePCD(pcd, 180)
            pcd.translate(np.array([x_max / 100.0, 0, 0]), relative = True)
            pcd.translate(np.array([-(w/2) / 100.0, (h/2) / 100.0, 0]), relative = True)
            pcd.translate(np.array([0, offset / 100.0, 0]), relative = True)
        elif(edge == 2): #T
            offset = x_max - L_max + random.choice([k for k in range(0, x_max - (x_max - L_max + h))])
            pcd = rotatePCD(pcd, -90)
            w, h = h, w
            pcd.translate(np.array([0, y_max / 100.0, 0]), relative = True)
            pcd.translate(np.array([(w/2) / 100.0, -(h/2) / 100.0, 0]), relative = True)
            pcd.translate(np.array([offset / 100.0, 0, 0]), relative = True)
        else:            #B
            offset = x_max - L_max + random.choice([k for k in range(0, x_max - (x_max - L_max + h))])
            pcd = rotatePCD(pcd, 90)
            w, h = h, w
            pcd.translate(np.array([(w/2) / 100.0, (h/2) / 100.0, 0]), relative = True)
            pcd.translate(np.array([offset / 100.0, 0, 0]), relative = True)
        
        pcds.append(pcd)
        print("\tEdge: " + str(edge) + "\tOffset: " + str(offset))
        
        overlap_count = 0
        array_label = -1
        if numObject in desk_list:
            array_label = numObject + 50
        else:
            array_label = numObject
        
        for (x, y, z) in pcd.points:
            try:
                xx = int(x * 100)
                yy = int(y * 100)
                if room[xx, yy] == array_label:
                    continue
                elif room[xx, yy] == 0:
                    room[xx, yy] = array_label
                else:
                    overlap_count += 1
                    if overlap_count > 500:
                        return False, None
            except:
                pass
            
    
    else:
        unit = w
        if unit < h:
            unit = h
            
        x = -1
        y = -1
            
        array_label = -1
        if numObject in desk_list:
            array_label = numObject + 50
            unit = int(unit / 2) + 1
            x = random.randint(unit + 5, room.shape[0] - unit - 5)
            y = random.randint(unit + 5, room.shape[1] - unit - 5)
        else:
            print('CHAIR')
            array_label = numObject + 80
            unit = int(unit / 2) + 1    
            score = 0
            for chair_try in range(0, 10):
                x_temp = random.randint(unit + 5, room.shape[0] - unit - 5)
                y_temp = random.randint(unit + 5, room.shape[1] - unit - 5)
                
                print('X_TEMP, Y_TEMP: ', x_temp, y_temp)
                score_temp = calc_score(x_temp, y_temp, unit, room)
                print('\t\tSCORE: ', score_temp)
                if score_temp > score:
                    x = x_temp
                    y = y_temp
                    score = score_temp
            if x == -1:
                print('x == -1')
                return False, None
        
        
        angles = [360, 90, 180, -90]
        angle = random.choice(angles)
        print("W, H, UNIT: ", w, h, unit)
        print("X, Y, ANGLE: ", x, y, angle)
        pcd = rotatePCD(pcd, angle)
        pcd.translate(np.array([x / 100.0, y / 100.0, 0]), relative = True)
    
        pcds.append(pcd)
    
        overlap_count = 0
        for (x, y, z) in pcd.points:
            xx = int(x * 100)
            yy = int(y * 100)
            if room[xx, yy] == array_label:
                continue
            elif room[xx, yy] == 0:
                room[xx, yy] = array_label
            elif room[xx, yy] > 80:
                continue
            else:
                overlap_count += 1
                if overlap_count > 500:
                    return False, None
    
    print_room(room)

    success, result = rec_place_objects(room, pcd_addrs, pcds, 5 + numObject, numObject+1, desk_list)
    if success == True:
        return True, result
    else:
        return rec_place_objects(_room, _pcd_addrs, _pcds, numTry-1, numObject, desk_list)
    

def calc_score(x, y, unit, room):
    score = 0
    for i in range(x - 2 * unit, x + 2 * unit):
        for j in range(y - 2 * unit, y + 2 * unit):
            try:
                #print('ROOM[I][J]: ', room[i][j])
                if int(room[i][j] / 10) == 5:
                    if (i > x - unit and i < x + unit) and (j > y - unit and j < y + unit):
                        score -= 10
                    else:
                        score += 5
                elif int(room[i][j] / 10) >= 8:
                    score -= 5
            except:
                pass
    return score
                        
    


def print_room(room):
    f = open("result.txt", 'a')
    i = room.shape[1]-1 # height
    j = 0        
    while i >= 0:
        if (i % 10 == 0 or i == room.shape[1]-1):
            j = 0
            while j < room.shape[0]: # width
                if (j % 10 == 0 or j == room.shape[0] - 1):
                    if(int(room[j,i]) == 0):
                        print('■', end='')
                        f.write('■')
                    else:
                        ee = ''
                        if int(room[j,i] < 10):
                           ee = ' '
                        print(int(room[j,i]), end=ee)
                        f.write(str(int(room[j,i])) + ee)
                j += 1
            print("\n", end="")
            f.write('\n')
        i -= 1
    print("\n", end="")
    f.write('\n')
    f.close()

def rotatePCD(pcd, angle):
    R = pcd.get_rotation_matrix_from_axis_angle((np.array([0,0, angle * math.pi / 180.0])))
    return pcd.rotate(R, center = (0,0,0))
            
def available_boundary(room, w, h):
    result = []
    result.append(boundary_length(room, (0,0),   (0, h-1)))
    result.append(boundary_length(room, (w-1,0), (w-1, h-1)))
    result.append(boundary_length(room, (0,h-1), (w-1, h-1)))
    result.append(boundary_length(room, (0,0),   (w-1, 0)))
    return result

def boundary_length(room, xy_min, xy_max):
    x_min, y_min = xy_min
    x_max, y_max = xy_max
    count = 0
    max_count = 0
    index = (x_min, y_min)
    for i in range (x_min, x_max+1):
        for j in range (y_min, y_max+1):
            if room[i, j] == 0:
                count += 1
                if count > max_count:
                    max_count = count
                    index = (i, j)
            else:
                count = 0
    return (max_count, index)


def unpack_data():
    s = []
    desk_list = []
    index = 1
    for target in ['11','12','8','9','10']:
        if isdir(join('fetched',target)) == True:
            files = list(reversed(os.listdir(join('fetched', target))))
            s = s + [join('fetched', target, f) for f in files]
            
            if target == '8':
                desk_list += [index + temp for temp in range(0, len(files))]
            index += len(files)
            
    return desk_list, s


if( __name__ == '__main__'):
    f = open("result.txt", 'w')
    f.close()
    rearrange()