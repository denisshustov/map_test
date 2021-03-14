import numpy as np
from numpy.lib.stride_tricks import as_strided
import collections

class Map_processor:
    def __init__(self):
        self.obsticle = 1
        self.visited_cell = 2

        self.setp = 0
        self.max_steps = 0
        self.x = 1
        self.y = 1
        self.set_default_extras()
        self.object_length_1 = 1
        self.object_length_2 = 2

        # self.a = np.array([ [1,1,1,1,1,1,1,1,1,1],
        #                     [1,0,0,0,1,1,0,0,0,1],
        #                     [1,0,0,0,1,1,0,0,0,1],
        #                     [1,0,0,0,1,1,0,0,0,1],
        #                     [1,0,0,0,1,1,0,0,0,1],
        #                     [1,0,0,0,0,0,0,0,0,1],
        #                     [1,0,0,0,0,0,0,0,0,1],
        #                     [1,0,0,0,0,0,0,0,0,1],
        #                     [1,0,0,0,1,0,0,0,0,1],
        #                     [1,1,1,1,1,0,0,0,1,1],
        #                     [1,0,0,0,0,0,0,0,0,1],
        #                     [1,0,0,0,0,0,0,0,0,1],
        #                     [1,0,0,0,0,0,0,0,0,1],
        #                     [1,1,1,1,1,1,1,1,1,1]])

        self.a = np.array([ [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                            [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],
                            [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                            [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                            [1,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1],
                            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
                            [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
                            [1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1],
                            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
                            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
                            [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        self.is_ob_general = False
        self.c_window = []
        self.apply_vars()
        self.max_steps = ((self.get_empty_cells_count() / len(self.c_window.flatten())) * 3)*3

    def set_default_extras(self):
        self.y_start_slice_extra = 0
        self.y_end_slice_extra = 0

        self.x_start_slice_extra = 0
        self.x_end_slice_extra = 0

    def get_empty_cells_count(self):
        return np.count_nonzero(self.a.flatten()==0)

    def set_visited(self):
        #self.success_area = (self.x,self.y,self.y_start_slice,self.y_end_slice, self.x_start_slice,self.x_end_slice)
        tmp_a = self.a[self.y_start_slice:self.y_end_slice, self.x_start_slice:self.x_end_slice]
        for i in range(0,9):
            if i==1:
                continue
            if len(self.a[self.y_start_slice:self.y_end_slice, self.x_start_slice:self.x_end_slice][tmp_a==i])>0:
                if i==0:
                    self.a[self.y_start_slice:self.y_end_slice, self.x_start_slice:self.x_end_slice][tmp_a==i]=i+2
                    break
                else:
                    self.a[self.y_start_slice:self.y_end_slice, self.x_start_slice:self.x_end_slice][tmp_a==i]=i+1
                    break

    def apply_vars(self):
        self.y_start_slice = self.y - self.object_length_1 + self.y_start_slice_extra
        self.y_end_slice = self.y + self.object_length_2 + self.y_end_slice_extra

        self.x_start_slice = self.x - self.object_length_1 + self.x_start_slice_extra
        self.x_end_slice = self.x + self.object_length_2 + self.x_end_slice_extra

        self.c_window = self.a[self.y_start_slice:self.y_end_slice,self.x_start_slice:self.x_end_slice]
        self.filled_obsticle_mask_array = np.full(self.c_window[:1].shape, self.obsticle)
        self.is_ob_general = self.obsticle in  self.c_window.flatten()
    
    def print_map(self):
        print(self.setp)
        self.apply_vars()
        tmp_a = np.array(self.a)
        tmp_a[self.y_start_slice:self.y_end_slice,self.x_start_slice:self.x_end_slice] = 9
        print(tmp_a)

    def try_move(self):
        self.apply_vars()
        if self.setp >= self.max_steps:
            return (False, False)

        move_variants = self.get_move_variants()
        if not move_variants is None:
            self.y += move_variants[0]
            self.x += move_variants[1]
            self.apply_vars()
            self.setp += 1
            return (True, True)
        
        print('No variants!!!')
        self.apply_vars()
        return (False, True)

    def get_diff(self, i, new_position):
        result = []
        if i[1]==-1:
            result.extend(new_position[:,:1].flatten())
        if i[1]==1:
            result.extend(new_position[:,new_position.shape[0]-1:].flatten())
        if i[0]==-1:
            result.extend(new_position[:1,:].flatten())
        if i[0]==1:
            result.extend(new_position[new_position.shape[0]-1:,:].flatten())

        return np.array(result)

    def get_move_variants(self):
        result = {False:{},True:{}}

        combination_of_all_elemets = np.array(np.meshgrid([-1,0,1],[-1,0,1])).T.reshape(-1,2)
        for i in combination_of_all_elemets:
            if np.array_equal(i,[0,0]):
                continue
        
            new_position = self.a[self.y_start_slice + i[0] : self.y_end_slice + i[0], self.x_start_slice + i[1] : self.x_end_slice + i[1]]
            if self.obsticle in new_position.flatten():
                continue

            diff = self.get_diff(i,new_position)
            setp_weight = np.average(diff.flatten()) 
            visited = (diff > 1).any()                   
            
            if new_position.shape[0] == self.c_window.shape[0] and new_position.shape[1] == self.c_window.shape[1]:
                if not setp_weight in result[visited]:
                    result[visited][setp_weight] = [i]
                else:
                    result[visited][setp_weight].append(i)

        # if len(result[False])==1:
        for k,v in collections.OrderedDict(sorted(result[False].items())).items():
            return v[0]
        # else:
        #     v=123
        # if len(result[True])==1:
        for k,v in collections.OrderedDict(sorted(result[True].items())).items():
            return v[0]
        # else:
        #     c=123
                
        return None

if __name__ == '__main__':
    mp = Map_processor()
    while True:
        mp.print_map()
        if(mp.get_empty_cells_count()==0):
            print('Done')
            break
        move_result = mp.try_move()
        if not(move_result[1]):
            print('reach max steps!!!')
            break
        if not move_result[0]:
            move_result = mp.try_move()
            
        if not mp.is_ob_general:
            mp.set_visited()            
        else:
            q=123
    t=123



# def try_extends(self, avoid_visited = False):
#     if self.is_ob_general:
#         return False

#     result = []
#     size = []
    
#     combination_of_all_elemets = np.array(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1])).T.reshape(-1,4)
#     for i in combination_of_all_elemets:
#         if np.array_equal(i,[0,0,0,0]):
#             continue
    
#         tmp_w = self.a[self.y_start_slice + i[0] : self.y_end_slice + i[1], self.x_start_slice + i[2] : self.x_end_slice + i[3]]
#         if tmp_w.shape[0] >= self.c_window.shape[0] and tmp_w.shape[1] >= self.c_window.shape[1]:
#             if self.obsticle not in tmp_w.flatten() and (not avoid_visited or (avoid_visited and self.visited_cell not in tmp_w.flatten())):
#                 result.append(i)
#                 size.append(tmp_w.shape[0]*tmp_w.shape[1])
    
#     if len(size):
#         max_index = np.argmax(size)
#         if result[max_index].any():
#             self.y_start_slice_extra += result[max_index][0]
#             self.y_end_slice_extra += result[max_index][1]
#             self.x_start_slice_extra += result[max_index][2]
#             self.x_end_slice_extra += result[max_index][3]
#             self.apply_vars()
#             self.try_extends(avoid_visited)


# def find_next_free_space(self):        
#     border = self.a[self.y_start_slice-1 : self.y_end_slice+1, self.x_start_slice-1 : self.x_end_slice+1]
#     self.set_default_extras()
#     zero_border_position = np.where(border == 0) 
#     #0-row, 1-column
#     if len(zero_border_position[0])>0:
#         i=0
#         while i < len(zero_border_position[0]):
#             self.y = zero_border_position[0][i]
#             self.x = zero_border_position[1][i]
#             self.try_move(True)

#             if not self.is_ob_general:
#                 break
#             i+=1
#         #self.try_extends(True)