## 📁 Project Setup & Architectural Design

### VS Code Folder Structure
The workspace is structured using a production-grade, modular design to isolate data operations, model logic, and evaluation routines:

<img width="292" height="768" alt="image" src="https://github.com/user-attachments/assets/4b511e2e-0ba5-4256-aaf9-0bd832b9c0eb" />

#### Architectural Breakdown:
*   **`data/raw/Images/`**: Direct landing directory housing the unsorted gray-scale surface defect `.jpg` images.
*   **`models/`**: Dedicated checkpoint directory storing optimized weight states (`.pth` files) after training execution.
*   **`src/`**: The logical core containing isolated functional scripts:
    *   `dataset.py`: Implements custom PyTorch pipeline transformations (`VisionInspectDataset`) for reliable tensor loading.
    *   `model.py`: Houses the CNN feature extraction and classification layer definitions.
    *   `prepare_data.py`: Handles string-parsing automation over raw filenames to extract target class domains.
    *   `train.py`: Contains the model training control loop, validation logic, and checkpoint serialization.

---

### 📊 Dataset Audit & Partition Integrity

The system automatically performs a localized runtime data verification audit on the raw folder. Filenames are decoded by splitting string boundaries (e.g., extracting the text prefix from `crazing_297.jpg`) to cleanly isolate categorical domains:

<img width="650" height="450" alt="image" src="https://github.com/user-attachments/assets/665f2ca1-7e0d-474f-b1d5-39fe86a2e684" />

#### Audit & Partition Metrics:
*   **Total Volume Discovered**: `1800` operational images across `6` strict surface defect classes.
*   **Class Balance Profile**: Highly balanced matrix ensuring `300` dedicated visual samples for every defect type (`crazing`, `inclusion`, `patches`, `pitted`, `rolled-in`, `scratches`). This avoids model prediction bias.
*   **Leakage-Free Splitting Engine**: To ensure robust optimization and strictly valid test metrics, data is mapped utilizing a `70/15/15` stratified split pipeline:
    *   **Train Split**: `1,260` images (used exclusively for backpropagation gradients).
    *   **Validation Split**: `270` images (used for hyperparameter tuning and model performance tracking).
    *   **Test Split**: `270` images (held out exclusively as an unseen baseline for final deployment evaluation).
