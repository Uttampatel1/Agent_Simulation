import pygame
from constants import (BUTTON_COLOR, BUTTON_HOVER_COLOR, BUTTON_TEXT_COLOR,
                       SELECTED_BUTTON_COLOR, WHITE)

class Button:
    """Represents a clickable button in the UI panel."""
    def __init__(self, x, y, w, h, text, action, font, tooltip=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.font = font
        self.is_hovered = False
        self.is_selected = False # For toggle buttons
        self.tooltip = tooltip
        self.tooltip_surf = None
        self.tooltip_rect = None
        if self.tooltip and font: # Pre-render tooltip
             self.tooltip_surf = font.render(self.tooltip, True, BUTTON_TEXT_COLOR, (0,0,0)) # Black background
             self.tooltip_rect = self.tooltip_surf.get_rect()

    def draw(self, screen):
        color = BUTTON_COLOR
        if self.is_selected: color = SELECTED_BUTTON_COLOR
        elif self.is_hovered: color = BUTTON_HOVER_COLOR

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 1) # Border

        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def draw_tooltip(self, screen, mouse_pos):
        """Draws the tooltip if hovered and tooltip exists."""
        if self.is_hovered and self.tooltip_surf:
            self.tooltip_rect.topleft = (mouse_pos[0] + 15, mouse_pos[1] + 10) # Position near cursor
            screen.blit(self.tooltip_surf, self.tooltip_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def handle_click(self):
        if self.is_hovered:
            return self.action
        return None

# Can add other UI element classes here (Sliders, etc.) later