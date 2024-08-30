import pygame
import sys
import math
import os

class SSVEPStimulus:
    """
    Class to handle the stimulus presentation paradigm for an SSVEP BCI system using flickering boxes.
    """
    
    def __init__(self, box_frequencies, refresh_rate=60, box_texts=None, box_text_indices=None, show_both=False, screen_resolution=None, display_index=0):
        """
        Initializes the SSVEPStimulus class.
        """
        if box_texts and len(box_texts) != len(box_text_indices):
            raise ValueError("The length of box_texts and box_text_indices must be the same if box_texts is provided.")

        pygame.init()

        # Get the list of available displays
        num_displays = pygame.display.get_num_displays()
        if display_index >= num_displays:
            raise ValueError(f"Display index {display_index} is out of range. There are only {num_displays} displays available.")
        
        desktop_sizes = pygame.display.get_desktop_sizes()
        selected_display_size = desktop_sizes[display_index]

        if screen_resolution:
            self.screen = pygame.display.set_mode(screen_resolution)
        else:
            # Set the position of the window to the selected display
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{selected_display_size[0]},0"
            self.screen = pygame.display.set_mode(selected_display_size, pygame.FULLSCREEN)

        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("Flickering Boxes")

        self.clock = pygame.time.Clock()

        if self.clock.get_fps():
            self.refresh_rate = self.clock.get_fps()
            print(f'Refresh Rate Measured at {self.refresh_rate} Hz')
        else:
            print(f"Could Not Measure Refresh Rate, Opting for Default or Input of: {refresh_rate} Hz")
            self.refresh_rate = refresh_rate

        sorted_indices = sorted(range(len(box_frequencies)), key=lambda i: box_frequencies[i])
        interleaved_indices = []

        # Interleave frequencies to maximize distance between similar frequencies
        left = 0
        right = len(sorted_indices) - 1
        while left <= right:
            if left == right:
                interleaved_indices.append(sorted_indices[left])
            else:
                interleaved_indices.append(sorted_indices[left])
                interleaved_indices.append(sorted_indices[right])
            left += 1
            right -= 1

        self.boxes = [{"rect": pygame.Rect(0, 0, 150, 150), "frequency": box_frequencies[i], "text": None, "frame_count": 0} for i in interleaved_indices]
        
        if box_texts and box_text_indices:
            for text, idx in zip(box_texts, box_text_indices):
                self.boxes[idx]["text"] = text
        
        self.background_color = pygame.Color('black')
        self.box_color = pygame.Color('white')
        
        self.start_button = pygame.Rect(self.screen_width // 2 - 50, self.screen_height // 2 - 25, 100, 50)
        self.start_button_color = pygame.Color('green')
        self.start_text = pygame.font.Font(None, 36).render('Start', True, pygame.Color('white'))
        self.start = False
        
        self.font = pygame.font.Font(None, 36)
        self.show_both = show_both

    def run(self):
        """
        Runs the main loop to handle the stimulus presentation.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.start_button.collidepoint(event.pos):
                        self.start = True

            self.screen.fill(self.background_color)

            if not self.start:
                pygame.draw.rect(self.screen, self.start_button_color, self.start_button)
                self.screen.blit(self.start_text, (self.start_button.x + 10, self.start_button.y + 10))
            else:
                centerX, centerY = self.screen_width // 2, self.screen_height // 2
                radius = min(self.screen_width, self.screen_height) // 3
                num_boxes = len(self.boxes)
                
                for i, box in enumerate(self.boxes):
                    angle = 2 * math.pi * i / num_boxes
                    box["rect"].center = (centerX + int(radius * math.cos(angle)), centerY + int(radius * math.sin(angle)))
                    
                    box["frame_count"] += 1
                    flicker_period = self.refresh_rate / box["frequency"]
                    if (box["frame_count"] % flicker_period) < (flicker_period / 2):
                        pygame.draw.rect(self.screen, self.box_color, box["rect"])
                        if self.show_both and box["text"]:
                            text_surface = self.font.render(box["text"], True, pygame.Color('black'))
                            text_rect = text_surface.get_rect(center=(box["rect"].centerx, box["rect"].centery - 10))
                            self.screen.blit(text_surface, text_rect)
                            
                            frequency_text = self.font.render(f"{box['frequency']} Hz", True, pygame.Color('black'))
                            frequency_rect = frequency_text.get_rect(center=(box["rect"].centerx, box["rect"].centery + 20))
                            self.screen.blit(frequency_text, frequency_rect)
                        else:
                            display_text = box["text"] if box["text"] else f"{box['frequency']} Hz"
                            text_surface = self.font.render(display_text, True, pygame.Color('black'))
                            text_rect = text_surface.get_rect(center=box["rect"].center)
                            self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            self.clock.tick(self.refresh_rate)

        pygame.quit()
        sys.exit()

# Example usage
if __name__ == "__main__":
    box_frequencies = [8, 10, 12, 14, 16, 18]  # List of frequencies
    box_texts = ["A", "B", "C"]  # List of texts or symbols
    box_text_indices = [0, 2, 4]  # Indices where the texts should be displayed
    refresh_rate = 60

    stimulus = SSVEPStimulus(box_frequencies, refresh_rate=refresh_rate, box_texts=box_texts, box_text_indices=box_text_indices, show_both=True, display_index=0)
    stimulus.run()
