import pygame
import math
from constants import QT_MAX_OBJECTS, QT_MAX_LEVELS, SIM_WIDTH, SIM_HEIGHT, DARK_GREY

class Quadtree:
    """ Basic Quadtree for 2D spatial partitioning. Objects need a 'rect' attribute. """
    def __init__(self, level, bounds):
        self.level = level
        self.bounds = bounds # pygame.Rect
        self.objects = []
        self.nodes = [None, None, None, None] # NW, NE, SW, SE

    def clear(self):
        self.objects = []
        for i in range(len(self.nodes)):
            if self.nodes[i] is not None:
                self.nodes[i].clear()
                self.nodes[i] = None # Explicitly remove reference

    def _split(self):
        sub_width = self.bounds.width / 2
        sub_height = self.bounds.height / 2
        x, y = self.bounds.x, self.bounds.y

        nw_bounds = pygame.Rect(x, y, math.ceil(sub_width), math.ceil(sub_height))
        ne_bounds = pygame.Rect(x + sub_width, y, math.floor(sub_width), math.ceil(sub_height))
        sw_bounds = pygame.Rect(x, y + sub_height, math.ceil(sub_width), math.floor(sub_height))
        se_bounds = pygame.Rect(x + sub_width, y + sub_height, math.floor(sub_width), math.floor(sub_height))

        self.nodes[0] = Quadtree(self.level + 1, nw_bounds)
        self.nodes[1] = Quadtree(self.level + 1, ne_bounds)
        self.nodes[2] = Quadtree(self.level + 1, sw_bounds)
        self.nodes[3] = Quadtree(self.level + 1, se_bounds)

    def _get_index(self, rect):
        index = -1
        if not self.bounds.contains(rect):
             return -1

        vert_mid = self.bounds.x + self.bounds.width / 2
        horz_mid = self.bounds.y + self.bounds.height / 2
        top_q = (rect.bottom <= horz_mid)
        bot_q = (rect.top >= horz_mid)
        left_q = (rect.right <= vert_mid)
        right_q = (rect.left >= vert_mid)

        if left_q:
            if top_q: index = 0 # NW
            elif bot_q: index = 2 # SW
        elif right_q:
            if top_q: index = 1 # NE
            elif bot_q: index = 3 # SE
        return index

    def insert(self, obj):
        if not hasattr(obj, 'rect'): return
        if not self.bounds.colliderect(obj.rect): return

        if self.nodes[0] is not None:
            index = self._get_index(obj.rect)
            if index != -1:
                self.nodes[index].insert(obj)
                return
        self.objects.append(obj)
        if len(self.objects) > QT_MAX_OBJECTS and self.level < QT_MAX_LEVELS:
            if self.nodes[0] is None: self._split()
            i = 0
            while i < len(self.objects):
                index = self._get_index(self.objects[i].rect)
                if index != -1:
                    self.nodes[index].insert(self.objects.pop(i))
                else: i += 1

    def query(self, query_rect):
        found = []
        found.extend([obj for obj in self.objects if query_rect.colliderect(obj.rect)])
        if self.nodes[0] is not None:
            for i in range(4):
                if self.nodes[i].bounds.colliderect(query_rect):
                    found.extend(self.nodes[i].query(query_rect))
        return found

    def query_radius(self, center_pos, radius):
        radius_sq = radius * radius
        query_bounds = pygame.Rect(center_pos[0] - radius, center_pos[1] - radius, radius * 2, radius * 2)
        query_bounds.clamp_ip(pygame.Rect(0, 0, SIM_WIDTH, SIM_HEIGHT))

        potential = self.query(query_bounds)
        nearby = []
        center_vec = pygame.Vector2(center_pos)
        for obj in potential:
            pos = getattr(obj, 'pos', None)
            if pos and isinstance(pos, pygame.Vector2):
                 if pos.distance_squared_to(center_vec) <= radius_sq:
                     nearby.append(obj)
            elif hasattr(obj, 'rect'): # Fallback to rect center
                 obj_center = pygame.Vector2(obj.rect.center)
                 if obj_center.distance_squared_to(center_vec) <= radius_sq:
                     nearby.append(obj)
        return nearby

    def draw(self, screen):
         pygame.draw.rect(screen, DARK_GREY, self.bounds, 1)
         if self.nodes[0] is not None:
             for node in self.nodes: node.draw(screen)