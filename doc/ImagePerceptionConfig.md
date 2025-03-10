# ImagePerceptionConfig

The `ap_gym.envs.image.ImagePerceptionConfig` class is a configuration class for image perception environments, such as
`ap_gym.envs.ImageClassification` and `ap_gym.envs.ImageLocalization`.
It holds the following parameters:

| Parameter                     | Type                                           | Default         | Description                                                                                                                                              |
|-------------------------------|------------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset`                     | `ap_gym.envs.image.ImageClassificationDataset` |                 | Dataset to use. Implemented types of datasets are `ap_gym.envs.image.CircleSquareDataset` and `ap_gym.envs.image.HuggingfaceImageClassificationDataset`. |
| `sensor_size`                 | `tuple[int, int]`                              | (5, 5)          | Size of the glimpse sensor in pixels.                                                                                                                    |
| `sensor_scale`                | `float`                                        | 1.0             | Relation of glimpse pixels to image pixels. A value of 2 means that glimpse pixels are twice as large as image pixels.                                   |
| `max_step_length`             | `float \| Sequence[float]`                     | 0.2             | Maximum normalized sensor movement per step relative to the total image size. Can be a single float or a sequence of floats for per-axis movement.       |
| `display_visitation`          | `bool`                                         | True            | Visualize glimpse visitation history during rendering.                                                                                                   |
| `render_overlay_base_color`   | `tuple[int, ...]`                              | (0, 0, 0, 0)    | Base color of the overlay used for unvisited areas in the image. The tuple represents RGBA values.                                                       |
| `render_good_color`           | `tuple[int, ...]`                              | (0, 255, 0, 80) | Color used in the overlay to indicate high prediction quality. The tuple represents RGBA values.                                                         |
| `render_bad_color`            | `tuple[int, ...]`                              | (255, 0, 0, 80) | Color used in the overlay to indicate low prediction quality. The tuple represents RGBA values.                                                          |
| `render_glimpse_shadow_color` | `tuple[int, ...]`                              | (0, 0, 0, 80)   | Color of the shadow drawn around the glimpse border during rendering. The tuple represents RGBA values.                                                  |
| `render_glimpse_border_color` | `tuple[int, ...]`                              | (0, 55, 255)    | Color of the border around the glimpse region during rendering. The tuple represents RGB values.                                                         |
| `prefetch_buffer_size`        | `int`                                          | 128             | Size of the prefetching buffer when using prefetching.                                                                                                   |
