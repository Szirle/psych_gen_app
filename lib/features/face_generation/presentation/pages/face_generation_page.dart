import 'dart:typed_data';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/foundation.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_event.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_state.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/characteristic_selector.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/custom_button.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/custom_number_text_field.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/dotted_background_painter.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/shimmer_image_placeholder.dart'
    as shimmer;
import 'dart:ui' as ui;
import 'package:psych_gen_app/features/face_generation/presentation/widgets/plotly_iframe_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/filters_panel.dart';
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

  void _logExpectedVsActual(String contextLabel, List<Uint8List> images) {
    try {
      final expected = _calculateImageCount();
      final actual = images.length;
      print('[FaceGenerationPage] ' +
          contextLabel +
          ' expected=' +
          expected.toString() +
          ' actual=' +
          actual.toString() +
          ' dims=' +
          faceManipulationRequest.manipulatedDimensions
              .map((d) => d.name.toString() + ':' + d.nLevels.toString())
              .join(','));
    } catch (e) {
      print('[FaceGenerationPage] logging error: ' + e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomPaint(
        painter: DottedBackgroundPainter(),
        child: Center(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Material(
                elevation: 10.0,
                child: Container(
                  width: 350,
                  padding: const EdgeInsets.all(10),
                  color: Colors.white,
                  child: ListView(
                    children: [
                      Row(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Tooltip(
                              message: 'tooltip.logo'.tr(),
                              child: Image.asset("assets/images/logo.png",
                                  width: 40, height: 40),
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
                        data: ThemeData()
                            .copyWith(dividerColor: Colors.transparent),
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
                              _buildReorderableSelectors(),
                              const SizedBox(height: 12),
                              _buildAddVariableButton(),
                              const SizedBox(height: 12),
                              _buildControlledVariablesSection(),
                            ]),
                          ],
                        ),
                      ),
                      // Filters section (closed by default)
                      Theme(
                        data: ThemeData()
                            .copyWith(dividerColor: Colors.transparent),
                        child: FiltersPanel(
                          currentDims:
                              faceManipulationRequest.manipulatedDimensions,
                        ),
                      ),
                      Theme(
                        data: ThemeData()
                            .copyWith(dividerColor: Colors.transparent),
                        child: ExpansionTile(
                          initiallyExpanded: true,
                          maintainState: true,
                          title: Text(
                            'section.settings'.tr(),
                            style: const TextStyle(
                              fontFamily: 'WorkSans',
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          children: <Widget>[
                            Padding(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 0, horizontal: 12),
                              child: Column(children: [
                                Text('settings.preserve_identity'.tr(),
                                    style: const TextStyle(fontSize: 12)),
                                Center(
                                    child: Tooltip(
                                  message: 'tooltip.preserve_identity'.tr(),
                                  child: Switch(
                                    value: faceManipulationRequest
                                        .preserveIdentity,
                                    onChanged: (value) {
                                      setState(() {
                                        faceManipulationRequest
                                            .preserveIdentity = value;
                                      });
                                      _loadImages();
                                    },
                                    activeColor: const Color(0xFF2B3A55),
                                  ),
                                )),
                                const SizedBox(height: 10),
                                Text('settings.truncation_psi'.tr(),
                                    style: const TextStyle(fontSize: 12)),
                                Row(
                                  children: [
                                    Expanded(
                                      child: Tooltip(
                                        message: 'tooltip.truncation_psi'.tr(),
                                        child: Slider(
                                          value: faceManipulationRequest
                                              .truncationPsi,
                                          min: 0.1,
                                          max: 1.0,
                                          divisions: 9,
                                          onChanged: (value) {
                                            setState(() {
                                              faceManipulationRequest
                                                  .truncationPsi = value;
                                            });
                                            _loadImages();
                                          },
                                        ),
                                      ),
                                    ),
                                    SizedBox(
                                      width: 50,
                                      child: Text(
                                        faceManipulationRequest.truncationPsi
                                            .toStringAsFixed(1),
                                        style: const TextStyle(fontSize: 16),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 10),
                                const SizedBox(height: 10),
                                Text('settings.mode'.tr(),
                                    style: const TextStyle(fontSize: 12)),
                                const SizedBox(height: 10),
                                SizedBox(
                                    height: 36,
                                    child: DropdownButtonFormField<String>(
                                      decoration: InputDecoration(
                                        focusedBorder: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(5.0),
                                          borderSide: const BorderSide(
                                              color: Colors.black26,
                                              width: 1.0),
                                        ),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(5.0),
                                          borderSide: const BorderSide(
                                              color: Colors.black26,
                                              width: 1.0),
                                        ),
                                        contentPadding: const EdgeInsets.only(
                                            top: 12, left: 12, right: 12),
                                      ),
                                      value: faceManipulationRequest.mode,
                                      onChanged: (String? newValue) {
                                        setState(() {
                                          faceManipulationRequest.mode =
                                              newValue!;
                                        });
                                        _loadImages();
                                      },
                                      items: ['shape', 'color', 'both']
                                          .map<DropdownMenuItem<String>>(
                                              (String value) {
                                        return DropdownMenuItem<String>(
                                          value: value,
                                          child: Text(
                                            value,
                                            style: const TextStyle(
                                              fontSize: 14,
                                              fontFamily: 'WorkSans',
                                            ),
                                          ),
                                        );
                                      }).toList(),
                                    )),
                                const SizedBox(height: 20),
                                Text('settings.num_images_each'.tr(),
                                    style: const TextStyle(fontSize: 12)),
                                const SizedBox(width: 20),
                                CustomNumberTextField(
                                    onChanged: (numberOfFaces) {
                                  setState(() {
                                    faceManipulationRequest.numFaces =
                                        numberOfFaces!;
                                  });
                                  _loadImages();
                                }),
                                Text('settings.total_images_info'.tr(),
                                    style: const TextStyle(fontSize: 12)),
                                const SizedBox(height: 20),
                                ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    elevation: 0,
                                    backgroundColor: const Color(0xFF2B3A55),
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 24, vertical: 18),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(5),
                                    ),
                                  ),
                                  onPressed: () {},
                                  child: Text(
                                    'button.generate_dataset'.tr(),
                                    style: const TextStyle(color: Colors.white),
                                  ),
                                ),
                              ]),
                            )
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Expanded(
                child: Column(
                  children: [
                    Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Padding(
                            padding: const EdgeInsets.fromLTRB(12, 0, 0, 0),
                            child: Text(
                              'preview.title'.tr(),
                              style: const TextStyle(
                                  fontFamily: 'WorkSans',
                                  fontSize: 28,
                                  color: Color(0xFF4A5568)),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.fromLTRB(12, 4, 0, 0),
                            child: Text(
                              'nav.breadcrumb'.tr(),
                              style: const TextStyle(
                                  fontFamily: 'WorkSans',
                                  fontSize: 11,
                                  color: Color(0xFF4A5568)),
                            ),
                          ),
                          Row(children: [
                            Padding(
                              padding: const EdgeInsets.fromLTRB(0, 8, 8, 0),
                              child: SizedBox(
                                width: 140,
                                child: Tooltip(
                                  message: 'tooltip.change_face'.tr(),
                                  child: ElevatedButton(
                                    style: ElevatedButton.styleFrom(
                                      elevation: 0,
                                      backgroundColor: const Color(0xFF2B3A55),
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 16, vertical: 12),
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(5),
                                      ),
                                    ),
                                    onPressed: () {
                                      final filtersPayload =
                                          _buildFiltersPayload();
                                      final immediateRequest =
                                          FaceManipulationRequest(
                                        manipulatedDimensions:
                                            faceManipulationRequest
                                                .manipulatedDimensions,
                                        truncationPsi: faceManipulationRequest
                                            .truncationPsi,
                                        numFaces:
                                            faceManipulationRequest.numFaces,
                                        preserveIdentity:
                                            faceManipulationRequest
                                                .preserveIdentity,
                                        mode: faceManipulationRequest.mode,
                                        changeFace: true,
                                        controlledVariables:
                                            faceManipulationRequest
                                                .controlledVariables,
                                        filters: filtersPayload.isEmpty
                                            ? null
                                            : filtersPayload,
                                      );
                                      context.read<FaceManipulationBloc>().add(
                                          LoadFaceImages(immediateRequest));
                                      _reloadCharts();
                                    },
                                    child: Text('button.change_face'.tr(),
                                        style: const TextStyle(
                                            color: Colors.white)),
                                  ),
                                ),
                              ),
                            ),
                          ])
                        ]),
                    Expanded(
                      child: Stack(
                        children: [
                          Positioned.fill(
                            child: BlocBuilder<FaceManipulationBloc,
                                FaceManipulationState>(
                              builder: (context, state) {
                                final dimensions = faceManipulationRequest
                                    .manipulatedDimensions;
                                final is3dMode = dimensions.length == 3 &&
                                    _xAxisDim != null &&
                                    _yAxisDim != null &&
                                    _sliderDim != null;
                                final is2dMode = dimensions.length == 2;

                                if (state is FaceManipulationLoading) {
                                  if (is3dMode) {
                                    return shimmer.ShimmerImagePlaceholder(
                                        rows: _yAxisDim!.nLevels,
                                        cols: _xAxisDim!.nLevels);
                                  } else if (is2dMode) {
                                    return shimmer.ShimmerImagePlaceholder(
                                        rows: dimensions[1].nLevels,
                                        cols: dimensions[0].nLevels);
                                  } else {
                                    return shimmer.ShimmerImagePlaceholder(
                                        count: _calculateImageCount());
                                  }
                                } else if (state is FaceManipulationLoaded) {
                                  if (is3dMode) {
                                    return Column(
                                      children: [
                                        _build3dSlider(),
                                        Expanded(
                                          child: InteractiveViewer(
                                            transformationController:
                                                _previewTransformController,
                                            minScale: 0.5,
                                            maxScale: 6.0,
                                            boundaryMargin: EdgeInsets.zero,
                                            child: LayoutBuilder(
                                              builder: (context, constraints) {
                                                return _build3dGridView(state,
                                                    constraints, dimensions);
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
                                        builder: (context, constraints) {
                                          return _build2dGridView(
                                              state, constraints, dimensions);
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
                                        builder: (context, constraints) {
                                          return _build1dRowView(
                                              state, constraints, dimensions);
                                        },
                                      ),
                                    );
                                  }
                                } else if (state is FaceManipulationError) {
                                  return shimmer.AnimatedImageWidget(
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Container(
                                          padding: const EdgeInsets.all(20),
                                          decoration: BoxDecoration(
                                            color: Colors.red[50],
                                            borderRadius:
                                                BorderRadius.circular(12),
                                            border: Border.all(
                                              color: Colors.red[200]!,
                                              width: 1,
                                            ),
                                          ),
                                          child: Column(
                                            children: [
                                              Icon(
                                                Icons.error_outline,
                                                color: Colors.red[400],
                                                size: 48,
                                              ),
                                              const SizedBox(height: 16),
                                              Text(
                                                'error.loading_images'.tr(),
                                                style: TextStyle(
                                                  color: Colors.red[700],
                                                  fontSize: 18,
                                                  fontWeight: FontWeight.bold,
                                                ),
                                              ),
                                              const SizedBox(height: 8),
                                              Text(
                                                state.message,
                                                style: TextStyle(
                                                  color: Colors.red[600],
                                                  fontSize: 14,
                                                ),
                                                textAlign: TextAlign.center,
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
                                      borderRadius: BorderRadius.circular(12),
                                      border: Border.all(
                                        color: Colors.grey[300]!,
                                        width: 1,
                                      ),
                                    ),
                                    child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(
                                          Icons.image_not_supported_outlined,
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
                              message: 'tooltip.recenter_preview'.tr(),
                              child: FloatingActionButton.small(
                                backgroundColor: const Color(0xFF2B3A55),
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
              SizedBox(
                width: 22,
                child: Center(
                  child: _buildChartsAnchor(isOpen: _showChartsPanel),
                ),
              ),
              if (_showChartsPanel)
                Container(
                  width: 380,
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
                        padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
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
            ],
          ),
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

  String _humanizeRaw(String s) {
    if (s.isEmpty) return s;
    final StringBuffer buffer = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      final String c = s[i];
      final bool isUpper = c.toUpperCase() == c && c.toLowerCase() != c;
      final bool prevIsLower = i > 0 && s[i - 1].toLowerCase() == s[i - 1];
      if (i > 0 && isUpper && prevIsLower) buffer.write(' ');
      buffer.write(c);
    }
    return buffer
        .toString()
        .split(' ')
        .map((w) => w.isEmpty ? w : (w[0].toUpperCase() + w.substring(1)))
        .join(' ');
  }

  String _labelForEnum(ManipulatedDimensionName name) =>
      _humanizeRaw(name.name);

  Widget _buildControlledVariablesSection() {
    final entries = ManipulatedDimensionName.values;
    return ExpansionTile(
      initiallyExpanded: false,
      maintainState: true,
      title: Text(
        'section.controlled_variables'.tr(),
        style: const TextStyle(
          fontFamily: 'WorkSans',
          fontWeight: FontWeight.bold,
        ),
      ),
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 6.0),
          child: Column(
            children: List.generate(entries.length, (index) {
              final name = entries[index];
              final bool checked = _selectedControlledVars.contains(name);
              return Column(
                children: [
                  CheckboxListTile(
                    dense: true,
                    controlAffinity: ListTileControlAffinity.leading,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 8.0),
                    activeColor: const Color(0xFF2B3A55),
                    value: checked,
                    onChanged: (bool? value) {
                      setState(() {
                        if (value == true) {
                          _selectedControlledVars.add(name);
                        } else {
                          _selectedControlledVars.remove(name);
                        }
                        faceManipulationRequest.controlledVariables =
                            _selectedControlledVars.isEmpty
                                ? null
                                : _selectedControlledVars.toList();
                      });
                      _loadImages();
                    },
                    title: Text(
                      _labelForEnum(name),
                      style: const TextStyle(
                        fontFamily: 'WorkSans',
                        fontSize: 12,
                      ),
                    ),
                  ),
                  if (index != entries.length - 1) const Divider(height: 1),
                ],
              );
            }),
          ),
        ),
      ],
    );
  }

  Widget _buildChartsAnchor({required bool isOpen}) {
    return Tooltip(
      message: isOpen ? 'panel.hide_charts'.tr() : 'panel.show_charts'.tr(),
      child: GestureDetector(
        onTap: () {
          setState(() {
            _showChartsPanel = !_showChartsPanel;
          });
        },
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          child: Container(
            width: 22,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(8),
                bottomLeft: Radius.circular(8),
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.08),
                  blurRadius: 6,
                  offset: const Offset(0, 2),
                ),
              ],
              border: Border.all(color: Colors.grey[300]!, width: 1),
            ),
            child: Transform.rotate(
              angle: isOpen ? math.pi : 0,
              child: const Icon(
                Icons.chevron_left,
                size: 18,
                color: Color(0xFF2B3A55),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _build3dSlider() {
    if (_sliderDim == null || _xAxisDim == null || _yAxisDim == null) {
      return const SizedBox.shrink();
    }

    return Container(
      padding: const EdgeInsets.all(16.0),
      margin: const EdgeInsets.only(bottom: 16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey[300]!, width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "${_sliderDim!.name.name} Level",
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  fontFamily: 'WorkSans',
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF2B3A55),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  "Level $_sliderValue",
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: const Color(0xFF2B3A55),
              inactiveTrackColor: Colors.grey[300],
              thumbColor: const Color(0xFF2B3A55),
              overlayColor: const Color(0xFF2B3A55).withOpacity(0.2),
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
              trackHeight: 4.0,
              valueIndicatorColor: const Color(0xFF2B3A55),
              valueIndicatorTextStyle: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
            child: Slider(
              value: _sliderValue.toDouble(),
              min: 1,
              max: _sliderDim!.nLevels.toDouble(),
              divisions: _sliderDim!.nLevels > 1 ? _sliderDim!.nLevels - 1 : 1,
              label: "Level $_sliderValue",
              onChanged: (newValue) {
                setState(() {
                  _sliderValue = newValue.round();
                });
              },
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "Level 1",
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.grey[600],
                ),
              ),
              Text(
                "Level ${_sliderDim!.nLevels}",
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAxisDropZones() {
    final dims = faceManipulationRequest.manipulatedDimensions;
    if (dims.isEmpty) {
      return Padding(
        padding: const EdgeInsets.all(8.0),
        child: Text('hint.add_variable_to_assign_axes'.tr(),
            textAlign: TextAlign.center),
      );
    }

    return Column(
      children: [
        if (dims.length == 1)
          _buildStaticAxisDisplay(dims.first, 'axis.x'.tr(), colors[0])
        else
          _buildAxisDropZone(
            axis: 'x',
            label: 'axis.x'.tr(),
            assignedDim: _xAxisDim,
            color: colors[0],
          ),
        if (dims.length > 1) const SizedBox(height: 8),
        if (dims.length > 1)
          _buildAxisDropZone(
            axis: 'y',
            label: 'axis.y'.tr(),
            assignedDim: _yAxisDim,
            color: colors[1],
          ),
        if (dims.length > 2) const SizedBox(height: 8),
        if (dims.length > 2)
          _buildAxisDropZone(
            axis: 'slider',
            label: 'axis.depth'.tr(),
            assignedDim: _sliderDim,
            color: colors[2],
          ),
      ],
    );
  }

  Widget _buildStaticAxisDisplay(
      ManipulatedDimension dim, String label, Color color) {
    return Container(
      height: 60,
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color, width: 2),
      ),
      child: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("$label: ",
                style: const TextStyle(fontWeight: FontWeight.bold)),
            _buildAssignedDimChip(dim, color),
          ],
        ),
      ),
    );
  }

  Widget _buildAxisDropZone({
    required String axis,
    required String label,
    required ManipulatedDimension? assignedDim,
    required Color color,
  }) {
    return DragTarget<ManipulatedDimension>(
      builder: (context, candidateData, rejectedData) {
        bool isTargeted = candidateData.isNotEmpty;
        Widget child;
        if (assignedDim != null) {
          final dimIndex = faceManipulationRequest.manipulatedDimensions
              .indexOf(assignedDim);
          child = _buildAssignedDimChip(
              assignedDim, dimIndex != -1 ? colors[dimIndex] : Colors.grey);
        } else {
          child = Center(
            child: Text(
              'axis.drop_here'.tr(namedArgs: {'label': label}),
              style: TextStyle(color: Colors.grey[600]),
            ),
          );
        }

        return Container(
          height: 60,
          width: double.infinity,
          decoration: BoxDecoration(
            color: isTargeted ? color.withOpacity(0.1) : Colors.grey[50],
            borderRadius: BorderRadius.circular(8),
          ),
          child: CustomPaint(
            painter: DottedBorderPainter(
              color: color,
              radius: const Radius.circular(8),
              strokeWidth: 2,
              gap: 4,
              dashWidth: 6,
            ),
            child: child,
          ),
        );
      },
      onWillAccept: (data) {
        if (data == null) return false;
        if (data == assignedDim) return false;
        return true;
      },
      onAccept: (data) {
        final newDim = data;
        final oldDimInTarget = assignedDim;

        String? sourceAxis;
        if (newDim == _xAxisDim) {
          sourceAxis = 'x';
        } else if (newDim == _yAxisDim) {
          sourceAxis = 'y';
        } else if (newDim == _sliderDim) {
          sourceAxis = 'slider';
        }

        setState(() {
          _setAxisValue(axis, newDim);
          if (sourceAxis != null) {
            _setAxisValue(sourceAxis, oldDimInTarget);
          }
        });
      },
    );
  }

  Widget _buildAssignedDimChip(ManipulatedDimension dim, Color color) {
    return Center(
      child: Chip(
        label: Text(dim.name.name,
            style: const TextStyle(
                color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: color,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      ),
    );
  }

  Widget _buildReorderableSelectors() {
    return Stack(
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 8.0, left: 8.0, right: 8.0),
          child: Column(
            children: List.generate(
                faceManipulationRequest.manipulatedDimensions.length, (index) {
              String label;
              if (index == 0) {
                label = "X-Axis";
              } else if (index == 1) {
                label = "Y-Axis";
              } else if (index == 2) {
                label = "Depth";
              } else {
                return const SizedBox.shrink();
              }

              return Container(
                height: 220,
                margin: const EdgeInsets.symmetric(vertical: 4),
                child: _buildAxisOutline(label: label, color: Colors.grey),
              );
            }),
          ),
        ),
        ReorderableListView(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          buildDefaultDragHandles: false,
          proxyDecorator: (child, index, animation) {
            return Material(
              elevation: 4.0,
              color: Colors.transparent,
              shadowColor: Colors.transparent,
              child: child,
            );
          },
          children: faceManipulationRequest.manipulatedDimensions
              .asMap()
              .entries
              .map((entry) {
            final index = entry.key;
            final dim = entry.value;
            return Padding(
              key: ValueKey(dim),
              padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
              child: Stack(
                children: [
                  CharacteristicSelector(
                    manipulatedDimension: dim,
                    allManipulatedDimensions:
                        faceManipulationRequest.manipulatedDimensions,
                    borderColor: _dimensionColors[dim] ?? Colors.grey,
                    onRangeChanged: (start, end) {
                      setState(() {
                        dim.rangeStart = start;
                        dim.rangeEnd = end;
                      });
                      _loadImages();
                    },
                    onCharacteristicSelected: (characteristicName) {
                      final isAlreadySelected = faceManipulationRequest
                          .manipulatedDimensions
                          .any((d) => d != dim && d.name == characteristicName);

                      if (isAlreadySelected) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                                '${characteristicName.name} is already selected.'),
                            backgroundColor: Colors.red,
                          ),
                        );
                      } else {
                        setState(() {
                          dim.name = characteristicName;
                        });
                        _loadImages();
                      }
                    },
                    onStrengthChanged: (strength) {
                      setState(() {
                        dim.strength = strength;
                      });
                      _loadImages();
                    },
                    onClose: () {
                      setState(() {
                        faceManipulationRequest.manipulatedDimensions
                            .remove(dim);
                        _updateDimensionColors();
                      });
                      _loadImages();
                    },
                    onNLevelChanged: (nLevel) {
                      setState(() {
                        dim.nLevels = nLevel;
                      });
                      _loadImages();
                    },
                  ),
                  Positioned(
                    top: 16,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: ReorderableDragStartListener(
                        index: index,
                        child: Container(
                          padding: const EdgeInsets.all(4),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(4),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.1),
                                blurRadius: 2,
                                offset: const Offset(0, 1),
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.drag_handle,
                            color: Colors.grey,
                            size: 16,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
          onReorder: (oldIndex, newIndex) {
            setState(() {
              if (oldIndex < newIndex) {
                newIndex -= 1;
              }
              final item = faceManipulationRequest.manipulatedDimensions
                  .removeAt(oldIndex);
              faceManipulationRequest.manipulatedDimensions
                  .insert(newIndex, item);
              _initOrUpdate3dState();
            });
          },
        ),
      ],
    );
  }

  Widget _buildAxisOutline({required String label, required Color color}) {
    final labelStyle = TextStyle(
      color: color,
      fontSize: 14,
      fontWeight: FontWeight.bold,
      fontFamily: 'WorkSans',
    );

    return Stack(
      clipBehavior: Clip.none,
      alignment: Alignment.topCenter,
      children: [
        Container(
          width: double.infinity,
          height: double.infinity,
          decoration: BoxDecoration(
            color: color.withOpacity(0.05),
            borderRadius: BorderRadius.circular(8),
          ),
          child: CustomPaint(
            painter: DottedBorderPainter(
              color: color,
              radius: const Radius.circular(8),
              strokeWidth: 2,
              gap: 4,
              dashWidth: 6,
            ),
          ),
        ),
        Transform.translate(
          offset: const Offset(0, -9),
          child: Container(
            color: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Tooltip(
              message: 'tooltip.axis_assignment'.tr(),
              child: Text(
                label,
                style: labelStyle,
              ),
            ),
          ),
        ),
      ],
    );
  }

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
                                  child: Image.memory(
                                    state.images[imageIndex],
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
                                  child: Image.memory(
                                    state.images[imageIndex],
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
                            child: Image.memory(
                              image,
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

class DottedBorderPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  final double gap;
  final double dashWidth;
  final Radius radius;

  DottedBorderPainter({
    this.color = Colors.black,
    this.strokeWidth = 1.0,
    this.gap = 5.0,
    this.dashWidth = 5.0,
    this.radius = const Radius.circular(0),
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    Path path = Path();
    path.addRRect(RRect.fromRectAndRadius(
        Rect.fromLTWH(0, 0, size.width, size.height), radius));

    ui.PathMetrics pathMetrics = path.computeMetrics();
    for (ui.PathMetric pathMetric in pathMetrics) {
      double distance = 0.0;
      while (distance < pathMetric.length) {
        canvas.drawPath(
          pathMetric.extractPath(distance, distance + dashWidth),
          paint,
        );
        distance += dashWidth + gap;
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

class AxisArrowsPainter extends CustomPainter {
  final ManipulatedDimension? xDim;
  final ManipulatedDimension? yDim;
  final ManipulatedDimension? zDim;
  final Map<ManipulatedDimension, Color> dimensionColors;

  AxisArrowsPainter({
    required this.xDim,
    required this.yDim,
    required this.zDim,
    required this.dimensionColors,
  });

  static const double _margin = 40.0;
  static const double _arrowThickness = 2.5;
  static const double _arrowHeadSize = 10.0;

  @override
  void paint(Canvas canvas, Size size) {
    final double left = _margin + 8;
    final double right = size.width - _margin - 56; // reserve for FAB
    final double bottom = size.height - _margin;
    final double top = _margin;

    if (xDim != null) {
      _drawArrow(
        canvas,
        start: Offset(left, bottom),
        end: Offset(right, bottom),
        color: dimensionColors[xDim] ?? const Color(0xFF4A90E2),
      );
      _drawCenteredLabel(
        canvas,
        text: xDim!.name.name,
        position: Offset((left + right) / 2, bottom + 16),
        color: dimensionColors[xDim] ?? const Color(0xFF4A90E2),
        rotateRadians: 0,
      );
    }

    if (yDim != null) {
      // Draw Y increasing upward
      _drawArrow(
        canvas,
        start: Offset(left, bottom),
        end: Offset(left, top),
        color: dimensionColors[yDim] ?? const Color(0xFFD53F8C),
      );
      _drawCenteredLabel(
        canvas,
        text: yDim!.name.name,
        position: Offset(left - 18, (top + bottom) / 2),
        color: dimensionColors[yDim] ?? const Color(0xFFD53F8C),
        rotateRadians: -math.pi / 2,
      );
    }

    if (zDim != null) {
      // Small diagonal depth arrow in the top-right corner
      final Offset start = Offset(right - 40, top + 8);
      final Offset end = Offset(right, top - 8 + 40);
      _drawArrow(
        canvas,
        start: start,
        end: end,
        color: dimensionColors[zDim] ?? const Color(0xFF3DBDBA),
      );
      _drawCenteredLabel(
        canvas,
        text: zDim!.name.name,
        position: Offset(end.dx + 4, end.dy),
        color: dimensionColors[zDim] ?? const Color(0xFF3DBDBA),
        rotateRadians: 0,
      );
    }
  }

  void _drawArrow(Canvas canvas,
      {required Offset start, required Offset end, required Color color}) {
    final Paint paint = Paint()
      ..color = color
      ..strokeWidth = _arrowThickness
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    // Shaft
    canvas.drawLine(start, end, paint);

    // Arrowhead
    final Offset direction = (end - start);
    final double length = direction.distance;
    if (length <= 0.001) return;
    final Offset unit = direction / length;

    // Perpendicular for head
    final Offset perp = Offset(-unit.dy, unit.dx);

    final Offset headBase = end - unit * _arrowHeadSize;
    final Offset p1 = end;
    final Offset p2 = headBase + perp * (_arrowHeadSize * 0.6);
    final Offset p3 = headBase - perp * (_arrowHeadSize * 0.6);

    final Path head = Path()
      ..moveTo(p1.dx, p1.dy)
      ..lineTo(p2.dx, p2.dy)
      ..lineTo(p3.dx, p3.dy)
      ..close();

    final Paint headPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    canvas.drawPath(head, headPaint);
  }

  void _drawCenteredLabel(Canvas canvas,
      {required String text,
      required Offset position,
      required Color color,
      required double rotateRadians}) {
    final TextPainter textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.w600,
          fontFamily: 'WorkSans',
        ),
      ),
      textDirection: ui.TextDirection.ltr,
    )..layout();

    final Size ts = textPainter.size;
    canvas.save();
    canvas.translate(position.dx, position.dy);
    canvas.rotate(rotateRadians);
    textPainter.paint(canvas, Offset(-ts.width / 2, -ts.height / 2));
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant AxisArrowsPainter oldDelegate) {
    return oldDelegate.xDim != xDim ||
        oldDelegate.yDim != yDim ||
        oldDelegate.zDim != zDim ||
        !mapEquals(oldDelegate.dimensionColors, dimensionColors);
  }
}

class AxisWrappingPainter extends CustomPainter {
  final ManipulatedDimension? xDim;
  final ManipulatedDimension? yDim;
  final ManipulatedDimension? zDim;
  final Map<ManipulatedDimension, Color> dimensionColors;

  AxisWrappingPainter({
    required this.xDim,
    required this.yDim,
    required this.zDim,
    required this.dimensionColors,
  });

  static const double _arrowThickness = 2.5;
  static const double _arrowHeadSize = 10.0;

  @override
  void paint(Canvas canvas, Size size) {
    final double left = 0;
    final double right = size.width;
    final double bottom = size.height;
    final double top = 0;

    if (xDim != null) {
      _drawArrow(canvas,
          start: Offset(left, bottom),
          end: Offset(right, bottom),
          color: dimensionColors[xDim] ?? const Color(0xFF4A90E2));
      _drawLabel(
        canvas,
        text: xDim!.name.name,
        position: Offset((left + right) / 2, bottom + 14),
        color: dimensionColors[xDim] ?? const Color(0xFF4A90E2),
        rotateRadians: 0,
      );
    }

    if (yDim != null) {
      _drawArrow(canvas,
          start: Offset(left, bottom),
          end: Offset(left, top),
          color: dimensionColors[yDim] ?? const Color(0xFFD53F8C));
      _drawLabel(
        canvas,
        text: yDim!.name.name,
        position: Offset(left - 18, (top + bottom) / 2),
        color: dimensionColors[yDim] ?? const Color(0xFFD53F8C),
        rotateRadians: -math.pi / 2,
      );
    }

    if (zDim != null) {
      final Offset start = Offset(right - 40, top + 8);
      final Offset end = Offset(right, top + 8 - 40);
      _drawArrow(canvas,
          start: start,
          end: end,
          color: dimensionColors[zDim] ?? const Color(0xFF3DBDBA));
      _drawLabel(
        canvas,
        text: zDim!.name.name,
        position: Offset(end.dx + 4, end.dy),
        color: dimensionColors[zDim] ?? const Color(0xFF3DBDBA),
        rotateRadians: 0,
      );
    }
  }

  void _drawArrow(Canvas canvas,
      {required Offset start, required Offset end, required Color color}) {
    final Paint paint = Paint()
      ..color = color
      ..strokeWidth = _arrowThickness
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(start, end, paint);

    final Offset direction = (end - start);
    final double length = direction.distance;
    if (length <= 0.001) return;
    final Offset unit = direction / length;
    final Offset perp = Offset(-unit.dy, unit.dx);
    final Offset headBase = end - unit * _arrowHeadSize;
    final Offset p1 = end;
    final Offset p2 = headBase + perp * (_arrowHeadSize * 0.6);
    final Offset p3 = headBase - perp * (_arrowHeadSize * 0.6);

    final Path head = Path()
      ..moveTo(p1.dx, p1.dy)
      ..lineTo(p2.dx, p2.dy)
      ..lineTo(p3.dx, p3.dy)
      ..close();

    final Paint headPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    canvas.drawPath(head, headPaint);
  }

  void _drawLabel(Canvas canvas,
      {required String text,
      required Offset position,
      required Color color,
      required double rotateRadians}) {
    final TextPainter textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.w600,
          fontFamily: 'WorkSans',
        ),
      ),
      textDirection: ui.TextDirection.ltr,
    )..layout();

    final Size ts = textPainter.size;
    canvas.save();
    canvas.translate(position.dx, position.dy);
    canvas.rotate(rotateRadians);
    textPainter.paint(canvas, Offset(-ts.width / 2, -ts.height / 2));
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant AxisWrappingPainter oldDelegate) {
    return oldDelegate.xDim != xDim ||
        oldDelegate.yDim != yDim ||
        oldDelegate.zDim != zDim ||
        !mapEquals(oldDelegate.dimensionColors, dimensionColors);
  }
}
