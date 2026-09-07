import 'dart:async';
import 'dart:math' as math;
import 'dart:convert';
import 'dart:typed_data';

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
import 'package:psych_gen_app/features/face_generation/domain/entities/full_resolution_face.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart'
    as shimmer;
import 'package:psych_gen_app/features/face_generation/presentation/widgets/filters_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_header_bar.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_painters.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/axis_assignment.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_grid_coordinates.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/controlled_variables_section.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/settings_panel.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/three_d_slider.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/filters_bloc.dart';
import 'package:psych_gen_app/features/face_generation/data/datasources/face_manipulation_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/quick_look_image_dialog.dart';
import 'package:psych_gen_app/core/designsystem/app_theme.dart';

class FaceGenerationPage extends StatefulWidget {
  const FaceGenerationPage({
    super.key,
    required this.title,
    this.traversalMode = false,
    this.numTraversals = 512,
    this.onThemeModeChanged,
  });

  final String title;
  final bool traversalMode;
  final int numTraversals;
  final ValueChanged<bool>? onThemeModeChanged;

  @override
  State<FaceGenerationPage> createState() => _FaceGenerationPageState();
}

class _FaceGenerationPageState extends State<FaceGenerationPage> {
  int _sliderValue = 1;
  ManipulatedDimension? _xAxisDim;
  ManipulatedDimension? _yAxisDim;
  ManipulatedDimension? _sliderDim;
  late final TransformationController _previewTransformController;
  // bool _showChartsPanel = true;
  // int _chartsReloadToken = 0;
  final Set<ManipulatedDimensionName> _selectedControlledVars = {};
  final FaceManipulationApiDataSource _faceApi =
      FaceManipulationApiDataSource();
  final Map<String, FullResolutionFace> _fullResolutionCache = {};
  final Map<String, Future<FullResolutionResponse>>
      _fullResolutionSessionStarts = {};
  final Map<String, Completer<FullResolutionFace>> _fullResolutionWaiters = {};
  final Map<String, String> _fullResolutionWaiterGridKeys = {};
  int _faceGeneration = 0;
  int _previewRevision = 0;
  int _fullResolutionPollGeneration = 0;
  String? _polledFullResolutionCacheKey;
  bool _quickLookOpen = false;
  bool _isPreviewCentered = true;

  List<Color> colors = [
    const Color(0xFF3DBDBA),
    const Color(0xFFD53F8C),
    const Color(0xFF4A90E2)
  ];

  final Map<ManipulatedDimension, Color> _dimensionColors = {};

  late final FaceManipulationRequest faceManipulationRequest;

  @override
  void initState() {
    super.initState();
    faceManipulationRequest = FaceManipulationRequest(
      manipulatedDimensions: [
        ManipulatedDimension(
          name: ManipulatedDimensionName.dominant,
          traversalIndex: widget.traversalMode ? 0 : null,
          strength: 25.0,
          nLevels: 2,
        )
      ],
      truncationPsi: 0.6,
      numFaces: 100,
      mode: 'both',
      preserveIdentity: false,
    );
    _previewTransformController = TransformationController();
    _previewTransformController.addListener(_handlePreviewTransformChanged);
    _updateDimensionColors();
    _initOrUpdate3dState();
    _loadImages();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _resetPreviewTransform();
    });
  }

  void _changeFace() {
    _invalidateQuickLookCache();
    final filtersPayload = _buildFiltersPayload();
    final immediateRequest = FaceManipulationRequest(
      manipulatedDimensions: faceManipulationRequest.manipulatedDimensions,
      truncationPsi: faceManipulationRequest.truncationPsi,
      numFaces: faceManipulationRequest.numFaces,
      preserveIdentity: faceManipulationRequest.preserveIdentity,
      mode: faceManipulationRequest.mode,
      previewRevision: _previewRevision,
      changeFace: true,
      controlledVariables: faceManipulationRequest.controlledVariables,
      filters: filtersPayload.isEmpty ? null : filtersPayload,
    );
    context.read<FaceManipulationBloc>().add(LoadFaceImages(immediateRequest));
  }

  void _loadImages({
    Map<ManipulatedDimensionName, List<double>>? filtersOverride,
  }) {
    _invalidateQuickLookCache();
    faceManipulationRequest.changeFace = false;
    final filtersPayload = filtersOverride == null
        ? _buildFiltersPayload()
        : _nonDefaultFilters(filtersOverride);
    faceManipulationRequest.filters =
        filtersPayload.isEmpty ? null : filtersPayload;
    faceManipulationRequest.controlledVariables =
        _selectedControlledVars.isEmpty
            ? null
            : _selectedControlledVars.toList();
    context
        .read<FaceManipulationBloc>()
        .add(LoadFaceImages(faceManipulationRequest));
    // _reloadCharts();
    _initOrUpdate3dState();
  }

  Map<ManipulatedDimensionName, List<double>> _buildFiltersPayload() {
    try {
      final fbState = context.read<FiltersBloc>().state;
      if (fbState is FiltersLoaded) {
        return _nonDefaultFilters(fbState.appliedFilters);
      }
    } catch (_) {}
    return {};
  }

  Map<ManipulatedDimensionName, List<double>> _nonDefaultFilters(
    Map<ManipulatedDimensionName, List<double>> filters,
  ) {
    final Map<ManipulatedDimensionName, List<double>> out = {};
    filters.forEach((key, range) {
      if (range.length >= 2) {
        final double start = range[0].clamp(0.0, 1.0);
        final double end = range[1].clamp(0.0, 1.0);
        if (!(start <= 0.0 && end >= 1.0)) {
          out[key] = [start, end];
        }
      }
    });
    return out;
  }

  // void _reloadCharts() {
  //   setState(() {
  //     _chartsReloadToken++;
  //   });
  // }

  void _resetPreviewTransform() {
    try {
      _previewTransformController.value = Matrix4.identity();
    } catch (_) {}
  }

  void _handlePreviewTransformChanged() {
    final values = _previewTransformController.value.storage;
    final scaleX = math.sqrt(values[0] * values[0] + values[1] * values[1]);
    final scaleY = math.sqrt(values[4] * values[4] + values[5] * values[5]);
    final translation = Offset(values[12], values[13]);

    // InteractiveViewer can introduce sub-pixel matrix changes during a click.
    // Treat those as centered so opening Quick Look does not reveal the pill.
    final isCentered = (scaleX - 1).abs() < 0.01 &&
        (scaleY - 1).abs() < 0.01 &&
        translation.distance < 3;
    if (isCentered != _isPreviewCentered && mounted) {
      setState(() => _isPreviewCentered = isCentered);
    }
  }

  Widget _buildInteractivePreviewCanvas({required Widget child}) {
    return MouseRegion(
      cursor: SystemMouseCursors.grab,
      child: InteractiveViewer(
        transformationController: _previewTransformController,
        panEnabled: true,
        scaleEnabled: true,
        panAxis: PanAxis.free,
        minScale: 0.5,
        maxScale: 6.0,
        boundaryMargin: const EdgeInsets.all(1000),
        trackpadScrollCausesScale: true,
        scaleFactor: 160,
        clipBehavior: Clip.hardEdge,
        child: child,
      ),
    );
  }

  Widget _buildDottedCanvas({
    required Widget child,
    required double spacing,
    required Offset origin,
  }) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Stack(
      fit: StackFit.expand,
      children: [
        CustomPaint(
          painter: DottedBackgroundPainter(
            color: isDark ? const Color(0xFF3D3F43) : const Color(0xFFC7CBD0),
            spacing: spacing,
            origin: origin,
          ),
        ),
        child,
      ],
    );
  }

  FaceManipulationRequest _snapshotFaceRequest() => FaceManipulationRequest(
        manipulatedDimensions: faceManipulationRequest.manipulatedDimensions
            .map((dimension) => ManipulatedDimension(
                  name: dimension.name,
                  traversalIndex: dimension.traversalIndex,
                  strength: dimension.strength,
                  nLevels: dimension.nLevels,
                  rangeStart: dimension.rangeStart,
                  rangeEnd: dimension.rangeEnd,
                ))
            .toList(),
        truncationPsi: faceManipulationRequest.truncationPsi,
        numFaces: faceManipulationRequest.numFaces,
        preserveIdentity: faceManipulationRequest.preserveIdentity,
        changeFace: false,
        mode: faceManipulationRequest.mode,
        previewRevision: faceManipulationRequest.previewRevision,
        filters: faceManipulationRequest.filters?.map(
          (key, value) => MapEntry(key, List<double>.from(value)),
        ),
        controlledVariables: faceManipulationRequest.controlledVariables == null
            ? null
            : List.of(faceManipulationRequest.controlledVariables!),
      );

  void _invalidateQuickLookCache() {
    final now = DateTime.now().microsecondsSinceEpoch;
    _previewRevision = math.max(now, _previewRevision + 1);
    faceManipulationRequest.previewRevision = _previewRevision;
    unawaited(
      _faceApi
          .cancelFullResolutionGeneration(_previewRevision)
          .catchError((Object _) {}),
    );
    _faceGeneration++;
    _fullResolutionPollGeneration++;
    _polledFullResolutionCacheKey = null;
    _fullResolutionCache.clear();
    _fullResolutionSessionStarts.clear();
    _fullResolutionWaiters.clear();
    _fullResolutionWaiterGridKeys.clear();
    if (_quickLookOpen && mounted) {
      Navigator.of(context, rootNavigator: true).pop();
      _quickLookOpen = false;
    }
  }

  String _fullResolutionGridKey(FaceManipulationRequest request) =>
      jsonEncode(request.toJson());

  String _fullResolutionEntryKey(
    FaceManipulationRequest request,
    List<int> selectedLevels,
  ) =>
      jsonEncode({
        ...request.toJson(),
        'selected_levels': selectedLevels,
      });

  FullResolutionFace? _cachedFullResolution(
    FaceManipulationRequest request,
    List<int> selectedLevels,
  ) =>
      _fullResolutionCache[_fullResolutionEntryKey(request, selectedLevels)];

  bool _sameLevels(List<int> first, List<int> second) {
    if (first.length != second.length) return false;
    for (var index = 0; index < first.length; index++) {
      if (first[index] != second[index]) return false;
    }
    return true;
  }

  void _storeFullResolution(
    FaceManipulationRequest request,
    List<int> selectedLevels,
    FullResolutionFace face,
  ) {
    final entryKey = _fullResolutionEntryKey(request, selectedLevels);
    _fullResolutionCache[entryKey] = face;
    final waiter = _fullResolutionWaiters.remove(entryKey);
    _fullResolutionWaiterGridKeys.remove(entryKey);
    if (waiter != null && !waiter.isCompleted) waiter.complete(face);
  }

  Future<FullResolutionFace> _waitForScheduledFullResolution(
    FaceManipulationRequest request,
    List<int> selectedLevels,
    String gridKey,
    String serverCacheKey,
  ) async {
    final entryKey = _fullResolutionEntryKey(request, selectedLevels);
    final cached = _fullResolutionCache[entryKey];
    if (cached != null) return cached;

    final existing = _fullResolutionWaiters[entryKey];
    if (existing != null) return existing.future;

    final waiter = Completer<FullResolutionFace>();
    _fullResolutionWaiters[entryKey] = waiter;
    _fullResolutionWaiterGridKeys[entryKey] = gridKey;
    try {
      // This endpoint only changes the order of already-scheduled work. It
      // never starts inference and is deliberately the only navigation call.
      await _faceApi.prioritizeFullResolution(
        serverCacheKey,
        selectedLevels,
      );
    } catch (error, stackTrace) {
      if (identical(_fullResolutionWaiters[entryKey], waiter)) {
        _fullResolutionWaiters.remove(entryKey);
        _fullResolutionWaiterGridKeys.remove(entryKey);
      }
      if (!waiter.isCompleted) waiter.completeError(error, stackTrace);
    }
    return waiter.future;
  }

  Future<FullResolutionFace> _requestFullResolution(
    FaceManipulationRequest request,
    List<int> selectedLevels,
    Uint8List previewBytes,
  ) {
    final entryKey = _fullResolutionEntryKey(request, selectedLevels);
    final cached = _fullResolutionCache[entryKey];
    if (cached != null) return Future.value(cached);

    final gridKey = _fullResolutionGridKey(request);
    var sessionStart = _fullResolutionSessionStarts[gridKey];
    final generation = _faceGeneration;
    if (sessionStart == null) {
      sessionStart = _faceApi.postFullResolutionFace(
        request,
        selectedLevels,
        previewBytes,
      );
      _fullResolutionSessionStarts[gridKey] = sessionStart;
    }

    final activeSessionStart = sessionStart;
    return activeSessionStart.then((response) async {
      if (generation != _faceGeneration) {
        throw StateError('The face changed during full-resolution generation.');
      }

      _storeFullResolution(
        request,
        response.selectedLevels,
        response.face,
      );
      if (mounted) setState(() {});

      if (response.previewIsLowResolution) {
        _startFullResolutionPolling(
          response.cacheKey,
          request,
          generation,
          gridKey,
        );
        final requested = _fullResolutionCache[entryKey];
        if (requested != null) return requested;
        return _waitForScheduledFullResolution(
          request,
          selectedLevels,
          gridKey,
          response.cacheKey,
        );
      }

      // A native 1024 preview requires no background image generation. Remove
      // the bootstrap slot so another coordinate may submit its preview solely
      // to prepare the matching latent. These calls remain serialized here.
      if (identical(
        _fullResolutionSessionStarts[gridKey],
        activeSessionStart,
      )) {
        _fullResolutionSessionStarts.remove(gridKey);
      }
      if (_sameLevels(response.selectedLevels, selectedLevels)) {
        return response.face;
      }
      return _requestFullResolution(request, selectedLevels, previewBytes);
    });
  }

  void _startFullResolutionPolling(
    String cacheKey,
    FaceManipulationRequest request,
    int faceGeneration,
    String gridKey,
  ) {
    if (_polledFullResolutionCacheKey == cacheKey) return;
    _polledFullResolutionCacheKey = cacheKey;
    final pollGeneration = ++_fullResolutionPollGeneration;
    Future<void>(() async {
      var cursor = 0;
      while (mounted &&
          faceGeneration == _faceGeneration &&
          pollGeneration == _fullResolutionPollGeneration) {
        try {
          final update =
              await _faceApi.fetchFullResolutionUpdates(cacheKey, cursor);
          // A response may arrive after a preview change or widget disposal.
          if (!mounted ||
              faceGeneration != _faceGeneration ||
              pollGeneration != _fullResolutionPollGeneration) {
            break;
          }
          cursor = update.cursor;
          if (update.items.isNotEmpty) {
            for (final item in update.items) {
              _storeFullResolution(
                request,
                item.selectedLevels,
                item.face,
              );
            }
            if (mounted) setState(() {});
          }
          if (update.complete) {
            final unresolved = _fullResolutionWaiterGridKeys.entries
                .where((entry) => entry.value == gridKey)
                .map((entry) => entry.key)
                .toList();
            for (final entryKey in unresolved) {
              final waiter = _fullResolutionWaiters.remove(entryKey);
              _fullResolutionWaiterGridKeys.remove(entryKey);
              if (waiter != null && !waiter.isCompleted) {
                waiter.completeError(
                  StateError(
                    update.error ??
                        'Full-resolution generation completed without this image.',
                  ),
                );
              }
            }
            break;
          }
        } catch (_) {
          // Keep the single status stream alive across transient HTTP errors.
        }
        await Future<void>.delayed(const Duration(milliseconds: 500));
      }
      if (pollGeneration == _fullResolutionPollGeneration) {
        _polledFullResolutionCacheKey = null;
      }
    });
  }

  Future<void> _openQuickLook(
    FaceImageGrid grid,
    List<ManipulatedDimension> dimensions,
    List<int> selectedLevels,
  ) async {
    final request = _snapshotFaceRequest();
    Uint8List? previewFor(List<int> levels) =>
        grid.imageFor(dimensions, levels);
    final previewBytes = previewFor(selectedLevels);
    if (previewBytes == null) return;
    final cached = _cachedFullResolution(request, selectedLevels);
    final fullResolutionFuture = cached == null
        ? _requestFullResolution(request, selectedLevels, previewBytes)
        : null;

    _quickLookOpen = true;
    await showGeneralDialog<void>(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'Close image quick look',
      barrierColor: Colors.black.withValues(alpha: 0.9),
      transitionDuration: const Duration(milliseconds: 120),
      pageBuilder: (context, animation, secondaryAnimation) =>
          QuickLookImageDialog(
        previewBytes: previewBytes,
        fullResolutionFace: cached,
        highResolutionFuture: fullResolutionFuture,
        initialSelectedLevels: selectedLevels,
        levelCounts: dimensions.map((dimension) => dimension.nLevels).toList(),
        horizontalAxis: math.max(
          0,
          dimensions.indexOf(_xAxisDim ?? dimensions.first),
        ),
        verticalAxis: dimensions.length < 2
            ? null
            : math.max(
                0,
                dimensions.indexOf(_yAxisDim ?? dimensions[1]),
              ),
        previewForLevels: previewFor,
        cachedFullResolutionForLevels: (levels) =>
            _cachedFullResolution(request, levels),
        requestFullResolutionForLevels: (levels, preview) =>
            _requestFullResolution(request, levels, preview),
      ),
      transitionBuilder: (context, animation, secondaryAnimation, child) =>
          FadeTransition(opacity: animation, child: child),
    );
    _quickLookOpen = false;
  }

  @override
  void dispose() {
    _fullResolutionPollGeneration++;
    _previewTransformController.removeListener(_handlePreviewTransformChanged);
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

  // _logExpectedVsActual removed (was unused)

  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    final double viewportHeight = mediaQuery.size.height;
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          final double viewportWidth =
              constraints.maxWidth.isFinite && constraints.maxWidth > 0
                  ? constraints.maxWidth
                  : mediaQuery.size.width;
          const double sidePanelWidth = 350;
          const double anchorWidth = 0;
          const double minPreviewWidth = 600;
          // const double chartsPanelWidth = 380;
          const double activeChartsWidth = 0;
          final double requiredWidth = sidePanelWidth +
              anchorWidth +
              activeChartsWidth +
              minPreviewWidth;
          final double effectiveWidth = math.max(viewportWidth, requiredWidth);

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
                          color: theme.colorScheme.surface,
                          child: ListView(
                            children: [
                              Row(
                                  crossAxisAlignment: CrossAxisAlignment.center,
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
                                              color: theme.colorScheme
                                                  .surfaceContainerHighest,
                                              borderRadius:
                                                  BorderRadius.circular(4),
                                            ),
                                            child: Icon(
                                              Icons.image_outlined,
                                              size: 24,
                                              color: theme
                                                  .colorScheme.onSurfaceVariant,
                                            ),
                                          );
                                        },
                                      ),
                                    ),
                                    const SizedBox(width: 5),
                                    Text(
                                      widget.title,
                                      style: TextStyle(
                                          fontFamily: 'WorkSans',
                                          fontWeight: FontWeight.bold,
                                          fontSize: 20,
                                          color: theme.colorScheme.onSurface),
                                    )
                                  ]),
                              const SizedBox(height: 32),
                              Theme(
                                data: theme.copyWith(
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
                                        traversalMode: widget.traversalMode,
                                        numTraversals: widget.numTraversals,
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
                                      if (!widget.traversalMode)
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
                              if (!widget.traversalMode)
                                Theme(
                                  data: theme.copyWith(
                                      dividerColor: Colors.transparent),
                                  child: FiltersPanel(
                                    currentDims: faceManipulationRequest
                                        .manipulatedDimensions,
                                    onFiltersCommitted: (filters) {
                                      _loadImages(filtersOverride: filters);
                                    },
                                  ),
                                ),
                              Theme(
                                data: theme.copyWith(
                                    dividerColor: Colors.transparent),
                                child: SettingsPanel(
                                  preserveIdentity:
                                      faceManipulationRequest.preserveIdentity,
                                  truncationPsi:
                                      faceManipulationRequest.truncationPsi,
                                  mode: faceManipulationRequest.mode,
                                  onPreserveIdentityChanged: (value) {
                                    setState(() {
                                      faceManipulationRequest.preserveIdentity =
                                          value;
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
                        child: ColoredBox(
                          color: isDark
                              ? AppTheme.darkCanvas
                              : AppTheme.lightCanvas,
                          child: Stack(
                            children: [
                              Positioned.fill(
                                child: BlocConsumer<FaceManipulationBloc,
                                    FaceManipulationState>(
                                  listener: (context, state) {
                                    if (state is FaceManipulationError &&
                                        state.previousGrid != null) {
                                      ScaffoldMessenger.of(context)
                                        ..hideCurrentSnackBar()
                                        ..showSnackBar(
                                          SnackBar(
                                            content: Text(state.message),
                                            backgroundColor: Colors.red[700],
                                          ),
                                        );
                                    }
                                  },
                                  builder: (context, state) {
                                    final dimensions = faceManipulationRequest
                                        .manipulatedDimensions;
                                    final is3dMode = dimensions.length == 3 &&
                                        _xAxisDim != null &&
                                        _yAxisDim != null &&
                                        _sliderDim != null;
                                    final is2dMode = dimensions.length == 2;
                                    final FaceImageGrid? grid = switch (state) {
                                      FaceManipulationLoaded(:final grid) =>
                                        grid,
                                      FaceManipulationLoading(
                                        :final previousGrid
                                      ) =>
                                        previousGrid,
                                      FaceManipulationError(
                                        :final previousGrid
                                      ) =>
                                        previousGrid,
                                      _ => null,
                                    };
                                    final isLoading =
                                        state is FaceManipulationLoading;

                                    if (grid != null || isLoading) {
                                      if (is3dMode) {
                                        return Column(
                                          children: [
                                            ThreeDLevelSlider(
                                              sliderDim: _sliderDim,
                                              sliderValue: _sliderValue
                                                  .clamp(
                                                    1,
                                                    _sliderDim!.nLevels,
                                                  )
                                                  .toInt(),
                                              onChanged: (val) {
                                                setState(() {
                                                  _sliderValue = val;
                                                });
                                              },
                                            ),
                                            Expanded(
                                              child:
                                                  _buildInteractivePreviewCanvas(
                                                child: LayoutBuilder(
                                                  builder:
                                                      (context, constraints) {
                                                    return _build3dGridView(
                                                      grid,
                                                      isLoading,
                                                      constraints,
                                                      dimensions,
                                                    );
                                                  },
                                                ),
                                              ),
                                            ),
                                          ],
                                        );
                                      } else if (is2dMode) {
                                        return _buildInteractivePreviewCanvas(
                                          child: LayoutBuilder(
                                            builder: (context, constraints) {
                                              return _build2dGridView(
                                                grid,
                                                isLoading,
                                                constraints,
                                                dimensions,
                                              );
                                            },
                                          ),
                                        );
                                      } else {
                                        return _buildInteractivePreviewCanvas(
                                          child: LayoutBuilder(
                                            builder: (context, constraints) {
                                              return _build1dRowView(
                                                grid,
                                                isLoading,
                                                constraints,
                                                dimensions,
                                              );
                                            },
                                          ),
                                        );
                                      }
                                    } else if (state is FaceManipulationError &&
                                        state.previousGrid == null) {
                                      return Column(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          Container(
                                            padding: const EdgeInsets.all(20),
                                            decoration: BoxDecoration(
                                              color: theme
                                                  .colorScheme.errorContainer,
                                              borderRadius:
                                                  BorderRadius.circular(12),
                                              border: Border.all(
                                                color: theme.colorScheme.error,
                                                width: 1,
                                              ),
                                            ),
                                            child: Column(
                                              children: [
                                                Icon(
                                                  Icons.error_outline,
                                                  color:
                                                      theme.colorScheme.error,
                                                  size: 48,
                                                ),
                                                const SizedBox(height: 16),
                                                Text(
                                                  'error.loading_images'.tr(),
                                                  style: TextStyle(
                                                    color: theme.colorScheme
                                                        .onErrorContainer,
                                                    fontSize: 18,
                                                    fontWeight: FontWeight.bold,
                                                  ),
                                                ),
                                                const SizedBox(height: 8),
                                                Text(
                                                  state.message,
                                                  style: TextStyle(
                                                    color: theme.colorScheme
                                                        .onErrorContainer,
                                                    fontSize: 14,
                                                  ),
                                                  textAlign: TextAlign.center,
                                                ),
                                              ],
                                            ),
                                          ),
                                        ],
                                      );
                                    }
                                    return Container(
                                      padding: const EdgeInsets.all(20),
                                      decoration: BoxDecoration(
                                        color:
                                            theme.colorScheme.surfaceContainer,
                                        borderRadius: BorderRadius.circular(12),
                                        border: Border.all(
                                          color:
                                              theme.colorScheme.outlineVariant,
                                          width: 1,
                                        ),
                                      ),
                                      child: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Icon(
                                            Icons.image_not_supported_outlined,
                                            color: theme
                                                .colorScheme.onSurfaceVariant,
                                            size: 48,
                                          ),
                                          const SizedBox(height: 16),
                                          Text(
                                            'grid.no_images'.tr(),
                                            style: TextStyle(
                                              color:
                                                  theme.colorScheme.onSurface,
                                              fontSize: 16,
                                            ),
                                          ),
                                          const SizedBox(height: 8),
                                          Text(
                                            'grid.adjust_settings'.tr(),
                                            style: TextStyle(
                                              color: theme
                                                  .colorScheme.onSurfaceVariant,
                                              fontSize: 12,
                                            ),
                                          ),
                                        ],
                                      ),
                                    );
                                  },
                                ),
                              ),
                              if (!_isPreviewCentered)
                                Positioned(
                                  left: 0,
                                  right: 0,
                                  bottom: 24,
                                  child: Center(
                                    child: FilledButton(
                                      key: const ValueKey(
                                        'recenter-preview-button',
                                      ),
                                      style: previewCanvasPillStyle(isDark),
                                      onPressed: _resetPreviewTransform,
                                      child: Text(
                                        'tooltip.recenter_preview'.tr(),
                                      ),
                                    ),
                                  ),
                                ),
                              Positioned(
                                left: 0,
                                top: 0,
                                right: 0,
                                child: PreviewHeaderBar(
                                  isDark: isDark,
                                  onThemeModeChanged: widget.onThemeModeChanged,
                                  onChangeFacePressed: _changeFace,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    // Charts panel temporarily disabled.
                    // SizedBox(
                    //   width: anchorWidth,
                    //   child: Center(
                    //     child: ChartsAnchor(
                    //       isOpen: _showChartsPanel,
                    //       onTap: () {
                    //         setState(() {
                    //           _showChartsPanel = !_showChartsPanel;
                    //         });
                    //       },
                    //     ),
                    //   ),
                    // ),
                    // if (_showChartsPanel)
                    //   SizedBox(
                    //     width: chartsPanelWidth,
                    //     child: Container(
                    //       height: double.infinity,
                    //       decoration: BoxDecoration(
                    //         color: Colors.white,
                    //         boxShadow: [
                    //           BoxShadow(
                    //             color: Colors.black.withOpacity(0.06),
                    //             blurRadius: 8,
                    //             offset: const Offset(0, 2),
                    //           ),
                    //         ],
                    //       ),
                    //       child: Column(
                    //         crossAxisAlignment: CrossAxisAlignment.start,
                    //         children: [
                    //           Padding(
                    //             padding:
                    //                 const EdgeInsets.fromLTRB(12, 8, 12, 4),
                    //             child: Text(
                    //               'panel.charts_title'.tr(),
                    //               style: const TextStyle(
                    //                 fontFamily: 'WorkSans',
                    //                 fontWeight: FontWeight.bold,
                    //                 fontSize: 16,
                    //                 color: Color(0xFF2B3A55),
                    //               ),
                    //             ),
                    //           ),
                    //           Expanded(
                    //             child: PlotlyIFramePanel(
                    //               reloadToken: _chartsReloadToken,
                    //             ),
                    //           ),
                    //         ],
                    //       ),
                    //     ),
                    //   ),
                  ],
                ),
              ),
            ),
          );
        },
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

          if (widget.traversalMode) {
            final selectedIndices = faceManipulationRequest
                .manipulatedDimensions
                .map((d) => d.traversalIndex)
                .whereType<int>()
                .toSet();
            final availableIndex = List.generate(widget.numTraversals, (i) => i)
                .firstWhere((i) => !selectedIndices.contains(i));
            faceManipulationRequest.manipulatedDimensions.add(
              ManipulatedDimension(
                name: ManipulatedDimensionName.dominant,
                traversalIndex: availableIndex,
                strength: 25.0,
                nLevels: 2,
              ),
            );
            setState(_updateDimensionColors);
            _loadImages();
            return;
          }

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

  Widget _build3dGridView(FaceImageGrid? grid, bool isLoading,
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
    final fullResolutionRequest = _snapshotFaceRequest();

    return _buildDottedCanvas(
      spacing: cellSize / 3,
      origin: Offset(
        (constraints.maxWidth - gridWidth) / 2,
        (constraints.maxHeight - gridHeight) / 2,
      ),
      child: Center(
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
                          // Flutter lays rows out from top to bottom, while the
                          // y-axis arrow denotes increasing values upward.
                          final yLevel = verticalLevelForDisplayRow(y, rows);
                          final s = _sliderValue
                                  .clamp(1, _sliderDim!.nLevels)
                                  .toInt() -
                              1;

                          final Map<ManipulatedDimension, int> levelMap = {
                            _xAxisDim!: x,
                            _yAxisDim!: yLevel,
                            _sliderDim!: s,
                          };

                          final level0 = levelMap[dimensions[0]]!;
                          final level1 = levelMap[dimensions[1]]!;
                          final level2 = levelMap[dimensions[2]]!;

                          final previewBytes = grid?.imageFor(
                            dimensions,
                            [level0, level1, level2],
                          );
                          final imageBytes = _cachedFullResolution(
                                fullResolutionRequest,
                                [level0, level1, level2],
                              )?.imageBytes ??
                              previewBytes;
                          return Padding(
                            padding: const EdgeInsets.all(itemPadding),
                            child: shimmer.AsyncGeneratedImageTile(
                              key: ValueKey('preview-3d-$x-$y'),
                              imageBytes: imageBytes,
                              size: imageSize,
                              isLoading: isLoading,
                              onTap: previewBytes == null ||
                                      isLoading ||
                                      grid == null
                                  ? null
                                  : () => _openQuickLook(
                                        grid,
                                        dimensions,
                                        [level0, level1, level2],
                                      ),
                            ),
                          );
                        }),
                      );
                    }),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _build2dGridView(FaceImageGrid? grid, bool isLoading,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final rows = dimensions[1].nLevels;
    final cols = dimensions[0].nLevels;
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
    final fullResolutionRequest = _snapshotFaceRequest();

    return _buildDottedCanvas(
      spacing: cellSize / 3,
      origin: Offset(
        (constraints.maxWidth - gridWidth) / 2,
        (constraints.maxHeight - gridHeight) / 2,
      ),
      child: Center(
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
                          // Display the largest second-dimension value in the
                          // top row so grid values follow the upward y arrow.
                          final yLevel = verticalLevelForDisplayRow(row, rows);
                          final previewBytes = grid?.imageFor(
                            dimensions,
                            [col, yLevel],
                          );
                          final imageBytes = _cachedFullResolution(
                                fullResolutionRequest,
                                [col, yLevel],
                              )?.imageBytes ??
                              previewBytes;
                          return Padding(
                            padding: const EdgeInsets.all(itemPadding),
                            child: shimmer.AsyncGeneratedImageTile(
                              key: ValueKey('preview-2d-$col-$row'),
                              imageBytes: imageBytes,
                              size: imageSize,
                              isLoading: isLoading,
                              onTap: previewBytes == null ||
                                      isLoading ||
                                      grid == null
                                  ? null
                                  : () => _openQuickLook(
                                        grid,
                                        dimensions,
                                        [col, yLevel],
                                      ),
                            ),
                          );
                        }),
                      );
                    }),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _build1dRowView(FaceImageGrid? grid, bool isLoading,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final imageCount = dimensions.isEmpty ? 1 : dimensions[0].nLevels;
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
    final fullResolutionRequest = _snapshotFaceRequest();

    return _buildDottedCanvas(
      spacing: cellSize / 3,
      origin: Offset(
        (constraints.maxWidth - gridWidth) / 2,
        (constraints.maxHeight - gridHeight) / 2,
      ),
      child: Center(
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
                    children: List.generate(imageCount, (index) {
                      final previewBytes = dimensions.isEmpty
                          ? null
                          : grid?.imageFor(dimensions, [index]);
                      final imageBytes = _cachedFullResolution(
                            fullResolutionRequest,
                            [index],
                          )?.imageBytes ??
                          previewBytes;
                      return Padding(
                        padding: const EdgeInsets.all(itemPadding),
                        child: shimmer.AsyncGeneratedImageTile(
                          key: ValueKey('preview-1d-$index'),
                          imageBytes: imageBytes,
                          size: imageSize,
                          isLoading: isLoading,
                          borderRadius: 8,
                          onTap:
                              previewBytes == null || isLoading || grid == null
                                  ? null
                                  : () => _openQuickLook(
                                        grid,
                                        dimensions,
                                        [index],
                                      ),
                        ),
                      );
                    }),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
