# psych_gen_app

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Proposed Architecture: Feature-Driven Clean Architecture with BLoC

Your current Flutter application combines UI, business logic, and data fetching within widgets, which can become difficult to manage. A more robust approach is a Feature-Driven Clean Architecture combined with the BLoC (Business Logic Component) pattern for state management. This architecture separates concerns into distinct layers, making the app easier to test, debug, and scale.

### Core Principles

- **Dependency Rule**: Inner layers should not know anything about outer layers. For example, the Domain layer should not have any dependency on the Presentation (UI) layer.
- **Feature-Driven Structure**: The code is organized by features (e.g., `face_generation`, `settings`, `dataset_export`) rather than by type (e.g., widgets, blocs, repositories). This makes it easier to find and work on all the code related to a specific feature.
- **BLoC for State Management**: The BLoC pattern decouples the UI from the business logic, handling user events and managing application state efficiently and predictably.

### Architectural Layers

- **Presentation Layer (UI)**
  - **Purpose**: Contains everything related to the UI, such as widgets and pages. This is the only layer that should have a dependency on Flutter's UI libraries (`material.dart`, `cupertino.dart`).
  - **Components**:
    - Pages/Screens: The main screen of your application (`HomePage`).
    - Widgets: Reusable UI components (e.g., `CharacteristicSelector`, `CustomButton`).
    - BLoCs/Cubits: Manages the state of the UI. It receives events from the UI, interacts with the Domain layer (via Use Cases), and emits new states to the UI.

- **Domain Layer (Business Logic)**
  - **Purpose**: This is the core of the application, containing the business logic and rules. It is completely independent of any other layer.
  - **Components**:
    - Entities/Models: Plain Dart objects representing the core data structures (e.g., `ManipulatedDimension`, `FaceManipulationRequest`).
    - Repositories (Abstract): Defines the contracts (interfaces) for data operations. The actual implementation is in the Data layer.
    - Use Cases: Encapsulates a single, specific business rule. For example, `GenerateFaceImagesUseCase`. It orchestrates the flow of data between the Presentation and Data layers.

- **Data Layer (Data Sources)**
  - **Purpose**: Responsible for retrieving data from various sources (API, local database, etc.).
  - **Components**:
    - Repositories (Implementation): Concrete implementations of the repository interfaces defined in the Domain layer.
    - Data Sources: Classes responsible for fetching raw data (e.g., `ApiService` for network requests).
    - Data Transfer Objects (DTOs): Models that represent the data coming from external sources. These are mapped to Domain layer entities.

### Proposed Directory Structure

```text
lib/
├── app/
│   ├── app.dart              # Main app widget, theme, routes
│   └── di/                   # Dependency Injection setup
│       └── service_locator.dart
│
├── core/
│   ├── error/                # Failure/Exception classes
│   │   ├── exceptions.dart
│   │   └── failures.dart
│   ├── usecases/             # Base use case class
│   │   └── usecase.dart
│   └── constants/            # App-wide constants (e.g., distributions.dart)
│       └── distributions.dart
│
├── features/
│   └── face_generation/
│       ├── data/
│       │   ├── datasources/
│       │   │   └── face_manipulation_api_datasource.dart # (was api_service.dart)
│       │   └── repositories/
│       │       └── face_manipulation_repository_impl.dart
│       │
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── manipulated_dimension.dart
│       │   │   └── face_manipulation_request.dart
│       │   ├── repositories/
│       │   │   └── face_manipulation_repository.dart # Abstract class
│       │   └── usecases/
│       │       └── generate_face_images.dart
│       │
│       └── presentation/
│           ├── bloc/
│           │   ├── face_manipulation_bloc.dart
│           │   ├── face_manipulation_event.dart
│           │   └── face_manipulation_state.dart
│           │
│           ├── pages/
│           │   └── home_page.dart
│           └── widgets/
│               ├── characteristic_selector.dart
│               ├── custom_button.dart
│               ├── distribution_range_selector.dart
│               └── ... # other widgets
│
└── main.dart                   # App entry point
```

### Library Suggestions

- **get_it**: A simple service locator for dependency injection. This helps in decoupling layers by providing instances of repositories and use cases where they are needed without tight coupling.
- **dartz**: For functional programming patterns, especially `Either`, to handle success and failure states gracefully from the Data layer up to the Presentation layer.
- **freezed**: A code generator for creating immutable data classes (entities/models), which helps prevent unintended state modifications.
- **dio** (optional, alternative to `http`): A more powerful HTTP client that supports interceptors, global configuration, and more.

### Cursor Rules for Architectural Adherence

To guide AI-assisted development toward this architecture, add rules under `.cursor/rules/` that enforce:

1. Feature-driven structure
2. Clean Architecture dependency boundaries
3. Standardized BLoC usage in presentation

