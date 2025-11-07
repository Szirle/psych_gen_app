import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_event.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_state.dart';
import 'package:psych_gen_app/core/designsystem/widgets/custom_button.dart';
// custom_number_text_field used via SettingsPanel
import 'package:psych_gen_app/core/designsystem/widgets/dotted_background_painter.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart'
    as shimmer;
import 'package:psych_gen_app/core/designsystem/widgets/safe_memory_image.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/plotly_iframe_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/filters_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_header_bar.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/charts_anchor.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_painters.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/axis_assignment.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/controlled_variables_section.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/settings_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/three_d_slider.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/filters_bloc.dart';

class FaceGenerationPage extends StatefulWidget {
  const FaceGenerationPage({super.key, required this.title});

  final String title;

  @override
  State<FaceGenerationPage> createState() => _FaceGenerationPageState();
}

class _FaceGenerationPageState extends State<FaceGenerationPage> {
  int _sliderValue = 1;
  ManipulatedDimension? _xAxisDim;
  ManipulatedDimension? _yAxisDim;
  ManipulatedDimension? _sliderDim;
  late final TransformationController _previewTransformController;
  bool _showChartsPanel = true;
  int _chartsReloadToken = 0;
  final Set<ManipulatedDimensionName> _selectedControlledVars = {};

  List<Color> colors = [
    const Color(0xFF3DBDBA),
    const Color(0xFFD53F8C),
    const Color(0xFF4A90E2)
  ];

  final Map<ManipulatedDimension, Color> _dimensionColors = {};

  FaceManipulationRequest faceManipulationRequest =
      FaceManipulationRequest(manipulatedDimensions: [
    ManipulatedDimension(
        name: ManipulatedDimensionName.dominant, strength: 25.0, nLevels: 2)
  ], truncationPsi: 0.6, numFaces: 100, mode: 'shape', preserveIdentity: false);

  @override
  void initState() {
    super.initState();
    _previewTransformController = TransformationController();
    _updateDimensionColors();
    _initOrUpdate3dState();
    _loadImages();
  }

  void _loadImages() {
    faceManipulationRequest.changeFace = false;
    final filtersPayload = _buildFiltersPayload();
    faceManipulationRequest.filters =
        filtersPayload.isEmpty ? null : filtersPayload;
    faceManipulationRequest.controlledVariables =
        _selectedControlledVars.isEmpty
            ? null
            : _selectedControlledVars.toList();
    context
        .read<FaceManipulationBloc>()
        .add(LoadFaceImages(faceManipulationRequest));
    _reloadCharts();
    _initOrUpdate3dState();
  }

  Map<ManipulatedDimensionName, List<double>> _buildFiltersPayload() {
    try {
      final fbState = context.read<FiltersBloc>().state;
      if (fbState is FiltersLoaded) {
        final Map<ManipulatedDimensionName, List<double>> out = {};
        fbState.appliedFilters.forEach((key, range) {
          if (range.length >= 2) {
            double start = range[0].clamp(0.0, 1.0);
            double end = range[1].clamp(0.0, 1.0);
            // Only include if not default [0,1]
            if (!(start <= 0.0 && end >= 1.0)) {
              out[key] = [start, end];
            }
          }
        });
        return out;
      }
    } catch (_) {}
    return {};
  }

  void _reloadCharts() {
    setState(() {
      _chartsReloadToken++;
    });
  }

  void _resetPreviewTransform() {
    try {
      _previewTransformController.value = Matrix4.identity();
    } catch (_) {}
  }

  @override
  void dispose() {
    _previewTransformController.dispose();
    super.dispose();
  }

  void _updateDimensionColors() {
    _dimensionColors.removeWhere((dim, color) =>
        !faceManipulationRequest.manipulatedDimensions.contains(dim));

    final assignedColors = _dimensionColors.values.toSet();
    final availableColors =
        colors.where((c) => !assignedColors.contains(c)).toList();

    for (var dim in faceManipulationRequest.manipulatedDimensions) {
      if (!_dimensionColors.containsKey(dim)) {
        if (availableColors.isNotEmpty) {
          _dimensionColors[dim] = availableColors.removeAt(0);
        } else {
          _dimensionColors[dim] = Colors.grey;
        }
      }
    }
  }

  void _initOrUpdate3dState() {
    final dims = faceManipulationRequest.manipulatedDimensions;

    setState(() {
      final newSliderDim = dims.length > 2 ? dims[2] : null;
      if (newSliderDim != _sliderDim) {
        _sliderValue = 1;
      }

      _xAxisDim = dims.isNotEmpty ? dims[0] : null;
      _yAxisDim = dims.length > 1 ? dims[1] : null;
      _sliderDim = newSliderDim;
    });
  }

  void _setAxisValue(String axis, ManipulatedDimension? newDim) {
    if (axis == 'x') {
      _xAxisDim = newDim;
    } else if (axis == 'y') {
      _yAxisDim = newDim;
    } else if (axis == 'slider') {
      _sliderDim = newDim;
      if (newDim != null) {
        _sliderValue = 1;
      }
    }
  }

  int _calculateImageCount() {
    int totalImages = 1;
    for (var dimension in faceManipulationRequest.manipulatedDimensions) {
      totalImages *= dimension.nLevels;
    }
    return totalImages;
  }

  // _logExpectedVsActual removed (was unused)

  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    final double viewportHeight = mediaQuery.size.height;

    return Scaffold(
      body: CustomPaint(
        painter: DottedBackgroundPainter(),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final double viewportWidth =
                constraints.maxWidth.isFinite && constraints.maxWidth > 0
                    ? constraints.maxWidth
                    : mediaQuery.size.width;
            const double sidePanelWidth = 350;
            const double anchorWidth = 22;
            const double minPreviewWidth = 600;
            const double chartsPanelWidth = 380;
            final double activeChartsWidth =
                _showChartsPanel ? chartsPanelWidth : 0;
            final double requiredWidth = sidePanelWidth +
                anchorWidth +
                activeChartsWidth +
                minPreviewWidth;
            final double effectiveWidth =
                math.max(viewportWidth, requiredWidth);

            return SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: ConstrainedBox(
                constraints: BoxConstraints(minWidth: effectiveWidth),
                child: SizedBox(
                  width: effectiveWidth,
                  height: viewportHeight,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: <Widget>[
                      Material(
                        elevation: 10.0,
                        child: SizedBox(
                          width: sidePanelWidth,
                          child: Container(
                            padding: const EdgeInsets.all(10),
                            color: Colors.white,
                            child: ListView(
                              children: [
                                Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.center,
                                    children: [
                                      Tooltip(
                                        message: 'tooltip.logo'.tr(),
                                        child: Image.asset(
                                          "assets/images/logo.png",
                                          width: 40,
                                          height: 40,
                                          errorBuilder:
                                              (context, error, stackTrace) {
                                            return Container(
                                              width: 40,
                                              height: 40,
                                              decoration: BoxDecoration(
                                                color: Colors.grey[200],
                                                borderRadius:
                                                    BorderRadius.circular(4),
                                              ),
                                              child: Icon(
                                                Icons.image_outlined,
                                                size: 24,
                                                color: Colors.grey[400],
                                              ),
                                            );
                                          },
                                        ),
                                      ),
                                      const SizedBox(width: 5),
                                      Text(
                                        'app.title'.tr(),
                                        style: const TextStyle(
                                            fontFamily: 'WorkSans',
                                            fontWeight: FontWeight.bold,
                                            fontSize: 20,
                                            color: Color(0xFF2B3A55)),
                                      )
                                    ]),
                                const SizedBox(height: 32),
                                Theme(
                                  data: ThemeData().copyWith(
                                      dividerColor: Colors.transparent),
                                  child: ExpansionTile(
                                    initiallyExpanded: true,
                                    maintainState: true,
                                    title: Text(
                                      'section.experimental_design'.tr(),
                                      style: const TextStyle(
                                        fontFamily: 'WorkSans',
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    children: <Widget>[
                                      Column(children: [
                                        AxisAssignment(
                                          manipulatedDimensions:
                                              faceManipulationRequest
                                                  .manipulatedDimensions,
                                          dimensionColors: _dimensionColors,
                                          xAxisDim: _xAxisDim,
                                          yAxisDim: _yAxisDim,
                                          sliderDim: _sliderDim,
                                          onAxisSet: (axis, dim) {
                                            setState(() {
                                              _setAxisValue(axis, dim);
                                            });
                                          },
                                          onDimsChanged: () {
                                            setState(() {});
                                            _loadImages();
                                          },
                                        ),
                                        const SizedBox(height: 12),
                                        _buildAddVariableButton(),
                                        const SizedBox(height: 12),
                                        ControlledVariablesSection(
                                          selectedControlledVars:
                                              _selectedControlledVars,
                                          onChanged: (name, checked) {
                                            setState(() {
                                              if (checked) {
                                                _selectedControlledVars
                                                    .add(name);
                                              } else {
                                                _selectedControlledVars
                                                    .remove(name);
                                              }
                                              faceManipulationRequest
                                                      .controlledVariables =
                                                  _selectedControlledVars
                                                          .isEmpty
                                                      ? null
                                                      : _selectedControlledVars
                                                          .toList();
                                            });
                                            _loadImages();
                                          },
                                        ),
                                      ]),
                                    ],
                                  ),
                                ),
                                // Filters section (closed by default)
                                Theme(
                                  data: ThemeData().copyWith(
                                      dividerColor: Colors.transparent),
                                  child: FiltersPanel(
                                    currentDims: faceManipulationRequest
                                        .manipulatedDimensions,
                                    onFiltersCommitted: () {
                                      _loadImages();
                                    },
                                  ),
                                ),
                                Theme(
                                  data: ThemeData().copyWith(
                                      dividerColor: Colors.transparent),
                                  child: SettingsPanel(
                                    preserveIdentity: faceManipulationRequest
                                        .preserveIdentity,
                                    truncationPsi:
                                        faceManipulationRequest.truncationPsi,
                                    mode: faceManipulationRequest.mode,
                                    onPreserveIdentityChanged: (value) {
                                      setState(() {
                                        faceManipulationRequest
                                            .preserveIdentity = value;
                                      });
                                      _loadImages();
                                    },
                                    onTruncationPsiChanged: (value) {
                                      setState(() {
                                        faceManipulationRequest.truncationPsi =
                                            value;
                                      });
                                      _loadImages();
                                    },
                                    onModeChanged: (newValue) {
                                      setState(() {
                                        faceManipulationRequest.mode = newValue;
                                      });
                                      _loadImages();
                                    },
                                    onNumFacesChanged: (numberOfFaces) {
                                      setState(() {
                                        faceManipulationRequest.numFaces =
                                            numberOfFaces;
                                      });
                                      _loadImages();
                                    },
                                    onGenerateDatasetPressed: () {},
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                      Expanded(
                        child: SizedBox(
                          height: viewportHeight,
                          child: Column(
                            children: [
                              PreviewHeaderBar(onChangeFacePressed: () {
                                final filtersPayload = _buildFiltersPayload();
                                final immediateRequest =
                                    FaceManipulationRequest(
                                  manipulatedDimensions: faceManipulationRequest
                                      .manipulatedDimensions,
                                  truncationPsi:
                                      faceManipulationRequest.truncationPsi,
                                  numFaces: faceManipulationRequest.numFaces,
                                  preserveIdentity:
                                      faceManipulationRequest.preserveIdentity,
                                  mode: faceManipulationRequest.mode,
                                  changeFace: true,
                                  controlledVariables: faceManipulationRequest
                                      .controlledVariables,
                                  filters: filtersPayload.isEmpty
                                      ? null
                                      : filtersPayload,
                                );
                                context
                                    .read<FaceManipulationBloc>()
                                    .add(LoadFaceImages(immediateRequest));
                                _reloadCharts();
                              }),
                              Expanded(
                                child: Stack(
                                  children: [
                                    Positioned.fill(
                                      child: BlocBuilder<FaceManipulationBloc,
                                          FaceManipulationState>(
                                        builder: (context, state) {
                                          final dimensions =
                                              faceManipulationRequest
                                                  .manipulatedDimensions;
                                          final is3dMode =
                                              dimensions.length == 3 &&
                                                  _xAxisDim != null &&
                                                  _yAxisDim != null &&
                                                  _sliderDim != null;
                                          final is2dMode =
                                              dimensions.length == 2;

                                          if (state
                                              is FaceManipulationLoading) {
                                            if (is3dMode) {
                                              return shimmer
                                                  .ShimmerImagePlaceholder(
                                                      rows: _yAxisDim!.nLevels,
                                                      cols: _xAxisDim!.nLevels);
                                            } else if (is2dMode) {
                                              return shimmer
                                                  .ShimmerImagePlaceholder(
                                                      rows:
                                                          dimensions[1].nLevels,
                                                      cols: dimensions[0]
                                                          .nLevels);
                                            } else {
                                              return shimmer
                                                  .ShimmerImagePlaceholder(
                                                      count:
                                                          _calculateImageCount());
                                            }
                                          } else if (state
                                              is FaceManipulationLoaded) {
                                            if (is3dMode) {
                                              return Column(
                                                children: [
                                                  ThreeDLevelSlider(
                                                    sliderDim: _sliderDim,
                                                    sliderValue: _sliderValue,
                                                    onChanged: (val) {
                                                      setState(() {
                                                        _sliderValue = val;
                                                      });
                                                    },
                                                  ),
                                                  Expanded(
                                                    child: InteractiveViewer(
                                                      transformationController:
                                                          _previewTransformController,
                                                      minScale: 0.5,
                                                      maxScale: 6.0,
                                                      boundaryMargin:
                                                          EdgeInsets.zero,
                                                      child: LayoutBuilder(
                                                        builder: (context,
                                                            constraints) {
                                                          return _build3dGridView(
                                                              state,
                                                              constraints,
                                                              dimensions);
                                                        },
                                                      ),
                                                    ),
                                                  ),
                                                ],
                                              );
                                            } else if (is2dMode) {
                                              return InteractiveViewer(
                                                transformationController:
                                                    _previewTransformController,
                                                minScale: 0.5,
                                                maxScale: 6.0,
                                                boundaryMargin: EdgeInsets.zero,
                                                child: LayoutBuilder(
                                                  builder:
                                                      (context, constraints) {
                                                    return _build2dGridView(
                                                        state,
                                                        constraints,
                                                        dimensions);
                                                  },
                                                ),
                                              );
                                            } else {
                                              return InteractiveViewer(
                                                transformationController:
                                                    _previewTransformController,
                                                minScale: 0.5,
                                                maxScale: 6.0,
                                                boundaryMargin: EdgeInsets.zero,
                                                child: LayoutBuilder(
                                                  builder:
                                                      (context, constraints) {
                                                    return _build1dRowView(
                                                        state,
                                                        constraints,
                                                        dimensions);
                                                  },
                                                ),
                                              );
                                            }
                                          } else if (state
                                              is FaceManipulationError) {
                                            return shimmer.AnimatedImageWidget(
                                              child: Column(
                                                mainAxisAlignment:
                                                    MainAxisAlignment.center,
                                                children: [
                                                  Container(
                                                    padding:
                                                        const EdgeInsets.all(
                                                            20),
                                                    decoration: BoxDecoration(
                                                      color: Colors.red[50],
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              12),
                                                      border: Border.all(
                                                        color: Colors.red[200]!,
                                                        width: 1,
                                                      ),
                                                    ),
                                                    child: Column(
                                                      children: [
                                                        Icon(
                                                          Icons.error_outline,
                                                          color:
                                                              Colors.red[400],
                                                          size: 48,
                                                        ),
                                                        const SizedBox(
                                                            height: 16),
                                                        Text(
                                                          'error.loading_images'
                                                              .tr(),
                                                          style: TextStyle(
                                                            color:
                                                                Colors.red[700],
                                                            fontSize: 18,
                                                            fontWeight:
                                                                FontWeight.bold,
                                                          ),
                                                        ),
                                                        const SizedBox(
                                                            height: 8),
                                                        Text(
                                                          state.message,
                                                          style: TextStyle(
                                                            color:
                                                                Colors.red[600],
                                                            fontSize: 14,
                                                          ),
                                                          textAlign:
                                                              TextAlign.center,
                                                        ),
                                                      ],
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            );
                                          }
                                          return shimmer.AnimatedImageWidget(
                                            child: Container(
                                              padding: const EdgeInsets.all(20),
                                              decoration: BoxDecoration(
                                                color: Colors.grey[100],
                                                borderRadius:
                                                    BorderRadius.circular(12),
                                                border: Border.all(
                                                  color: Colors.grey[300]!,
                                                  width: 1,
                                                ),
                                              ),
                                              child: Column(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(
                                                    Icons
                                                        .image_not_supported_outlined,
                                                    color: Colors.grey[400],
                                                    size: 48,
                                                  ),
                                                  const SizedBox(height: 16),
                                                  Text(
                                                    'grid.no_images'.tr(),
                                                    style: TextStyle(
                                                      color: Colors.grey[600],
                                                      fontSize: 16,
                                                    ),
                                                  ),
                                                  const SizedBox(height: 8),
                                                  Text(
                                                    'grid.adjust_settings'.tr(),
                                                    style: TextStyle(
                                                      color: Colors.grey[500],
                                                      fontSize: 12,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          );
                                        },
                                      ),
                                    ),
                                    Positioned(
                                      right: 24,
                                      bottom: 24,
                                      child: Tooltip(
                                        message:
                                            'tooltip.recenter_preview'.tr(),
                                        child: FloatingActionButton.small(
                                          backgroundColor:
                                              const Color(0xFF2B3A55),
                                          onPressed: _resetPreviewTransform,
                                          child: const Icon(
                                            Icons.center_focus_strong,
                                            color: Colors.white,
                                          ),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      SizedBox(
                        width: anchorWidth,
                        child: Center(
                          child: ChartsAnchor(
                            isOpen: _showChartsPanel,
                            onTap: () {
                              setState(() {
                                _showChartsPanel = !_showChartsPanel;
                              });
                            },
                          ),
                        ),
                      ),
                      if (_showChartsPanel)
                        SizedBox(
                          width: chartsPanelWidth,
                          child: Container(
                            height: double.infinity,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.06),
                                  blurRadius: 8,
                                  offset: const Offset(0, 2),
                                ),
                              ],
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Padding(
                                  padding:
                                      const EdgeInsets.fromLTRB(12, 8, 12, 4),
                                  child: Text(
                                    'panel.charts_title'.tr(),
                                    style: const TextStyle(
                                      fontFamily: 'WorkSans',
                                      fontWeight: FontWeight.bold,
                                      fontSize: 16,
                                      color: Color(0xFF2B3A55),
                                    ),
                                  ),
                                ),
                                Expanded(
                                  child: PlotlyIFramePanel(
                                    reloadToken: _chartsReloadToken,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildAddVariableButton() {
    return CustomElevatedButton(
      onPressed: () {
        if (faceManipulationRequest.manipulatedDimensions.length < 3) {
          final selectedNames = faceManipulationRequest.manipulatedDimensions
              .map((d) => d.name)
              .toSet();

          ManipulatedDimensionName? availableName;
          for (var name in ManipulatedDimensionName.values) {
            if (!selectedNames.contains(name)) {
              availableName = name;
              break;
            }
          }

          if (availableName != null) {
            faceManipulationRequest.manipulatedDimensions.add(
              ManipulatedDimension(
                  name: availableName, strength: 25.0, nLevels: 2),
            );
            setState(() {
              _updateDimensionColors();
            });
            _loadImages();
          }
        }
      },
      buttonText: 'button.add_variable'.tr(),
    );
  }

  // _build3dSlider removed (extracted)

  Widget _build3dGridView(FaceManipulationLoaded state,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final rows = _yAxisDim!.nLevels;
    final cols = _xAxisDim!.nLevels;
    const itemPadding = 4.0;
    const outerPadding = 8.0;

    final availableImageWidth =
        (constraints.maxWidth - outerPadding * 2 - (itemPadding * 2 * cols)) /
            cols;
    final availableImageHeight =
        (constraints.maxHeight - outerPadding * 2 - (itemPadding * 2 * rows)) /
            rows;
    final imageSize = (availableImageWidth < availableImageHeight
            ? availableImageWidth
            : availableImageHeight)
        .clamp(30.0, 150.0);

    final double cellSize = imageSize + itemPadding * 2;
    final double gridWidth = cols * cellSize;
    final double gridHeight = rows * cellSize;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(outerPadding),
        child: SizedBox(
          width: gridWidth,
          height: gridHeight,
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: AxisWrappingPainter(
                    xDim: _xAxisDim,
                    yDim: _yAxisDim,
                    zDim: null,
                    dimensionColors: _dimensionColors,
                  ),
                ),
              ),
              Align(
                alignment: Alignment.topLeft,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: List.generate(rows, (y) {
                    return Row(
                      mainAxisSize: MainAxisSize.min,
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: List.generate(cols, (x) {
                        final s = _sliderValue - 1;

                        final Map<ManipulatedDimension, int> levelMap = {
                          _xAxisDim!: x,
                          _yAxisDim!: y,
                          _sliderDim!: s,
                        };

                        final level0 = levelMap[dimensions[0]]!;
                        final level1 = levelMap[dimensions[1]]!;
                        final level2 = levelMap[dimensions[2]]!;

                        final nLevels0 = dimensions[0].nLevels;
                        final nLevels1 = dimensions[1].nLevels;

                        final imageIndex = level2 * (nLevels1 * nLevels0) +
                            level1 * nLevels0 +
                            level0;

                        if (imageIndex < state.images.length) {
                          return Padding(
                            padding: const EdgeInsets.all(itemPadding),
                            child: shimmer.AnimatedImageWidget(
                              child: Container(
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(6),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.1),
                                      blurRadius: 4,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(6),
                                  child: SafeMemoryImage(
                                    imageBytes: state.images[imageIndex],
                                    width: imageSize,
                                    height: imageSize,
                                    fit: BoxFit.cover,
                                  ),
                                ),
                              ),
                            ),
                          );
                        } else {
                          return SizedBox(
                              width: imageSize + (itemPadding * 2),
                              height: imageSize + (itemPadding * 2));
                        }
                      }),
                    );
                  }),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _build2dGridView(FaceManipulationLoaded state,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final isFiveByFive = dimensions.length == 2 &&
        dimensions[0].nLevels == 5 &&
        dimensions[1].nLevels == 5 &&
        state.images.length >= 9;

    final rows = isFiveByFive ? 5 : dimensions[1].nLevels;
    final cols = isFiveByFive ? 5 : dimensions[0].nLevels;
    const itemPadding = 4.0;
    const outerPadding = 8.0;

    final availableImageWidth =
        (constraints.maxWidth - outerPadding * 2 - (itemPadding * 2 * cols)) /
            cols;
    final availableImageHeight =
        (constraints.maxHeight - outerPadding * 2 - (itemPadding * 2 * rows)) /
            rows;
    final imageSize = (availableImageWidth < availableImageHeight
            ? availableImageWidth
            : availableImageHeight)
        .clamp(30.0, 150.0);

    final double cellSize = imageSize + itemPadding * 2;
    final double gridWidth = cols * cellSize;
    final double gridHeight = rows * cellSize;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(outerPadding),
        child: SizedBox(
          width: gridWidth,
          height: gridHeight,
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: AxisWrappingPainter(
                    xDim: dimensions.isNotEmpty ? dimensions[0] : null,
                    yDim: dimensions.length > 1 ? dimensions[1] : null,
                    zDim: null,
                    dimensionColors: _dimensionColors,
                  ),
                ),
              ),
              Align(
                alignment: Alignment.topLeft,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: List.generate(rows, (row) {
                    return Row(
                      mainAxisSize: MainAxisSize.min,
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: List.generate(cols, (col) {
                        int imageIndex = -1;
                        if (isFiveByFive) {
                          const int middleRow = 2;
                          const int middleCol = 2;
                          if (row == middleRow) {
                            imageIndex = col;
                          } else if (col == middleCol) {
                            imageIndex = 5 + row;
                          }
                        } else {
                          imageIndex = row * cols + col;
                        }

                        if (imageIndex != -1 &&
                            imageIndex < state.images.length) {
                          return Padding(
                            padding: const EdgeInsets.all(itemPadding),
                            child: shimmer.AnimatedImageWidget(
                              child: Container(
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(6),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.1),
                                      blurRadius: 4,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(6),
                                  child: SafeMemoryImage(
                                    imageBytes: state.images[imageIndex],
                                    width: imageSize,
                                    height: imageSize,
                                    fit: BoxFit.cover,
                                  ),
                                ),
                              ),
                            ),
                          );
                        } else {
                          return SizedBox(
                              width: imageSize + (itemPadding * 2),
                              height: imageSize + (itemPadding * 2));
                        }
                      }),
                    );
                  }),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _build1dRowView(FaceManipulationLoaded state,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final imageCount = state.images.length;
    const double itemPadding = 8.0;
    const double outerPadding = 8.0;

    final availableImageWidth = constraints.maxWidth -
        outerPadding * 2 -
        ((itemPadding * 2) * imageCount);
    final calculatedImageSize = availableImageWidth / imageCount;
    final imageSize = calculatedImageSize.clamp(20.0, 200.0);

    final double cellSize = imageSize + itemPadding * 2;
    final double gridWidth = imageCount * cellSize;
    final double gridHeight = cellSize;

    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: outerPadding),
        child: SizedBox(
          width: gridWidth,
          height: gridHeight,
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: AxisWrappingPainter(
                    xDim: dimensions.isNotEmpty ? dimensions[0] : null,
                    yDim: null,
                    zDim: null,
                    dimensionColors: _dimensionColors,
                  ),
                ),
              ),
              Align(
                alignment: Alignment.topLeft,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: state.images.map((image) {
                    return Padding(
                      padding: const EdgeInsets.all(itemPadding),
                      child: shimmer.AnimatedImageWidget(
                        child: Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(8),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.1),
                                blurRadius: 8,
                                offset: const Offset(0, 4),
                              ),
                            ],
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: SafeMemoryImage(
                              imageBytes: image,
                              width: imageSize,
                              height: imageSize,
                              fit: BoxFit.cover,
                            ),
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
