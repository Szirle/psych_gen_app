import 'dart:math' as math;
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/core/designsystem/app_theme.dart';
import 'package:psych_gen_app/core/designsystem/widgets/dotted_background_painter.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart';
import '../../../domain/entities/manipulated_dimension_name.dart';
import '../../bloc/stimuli_selection_cubit.dart';
import '../filters_panel.dart';
import 'preview_header_bar.dart';

class StimuliSelectionSettings extends StatelessWidget {
  final double truncationPsi;
  final ValueChanged<double> onTruncationChanged;
  final ValueChanged<Map<ManipulatedDimensionName, List<double>>>
      onFiltersCommitted;

  const StimuliSelectionSettings(
      {super.key,
      required this.truncationPsi,
      required this.onTruncationChanged,
      required this.onFiltersCommitted});

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
              padding: const EdgeInsets.all(12),
              child: Text('selection.description'.tr())),
          Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Text('settings.truncation_psi'.tr())),
          Row(children: [
            Expanded(
                child: Slider(
                    value: truncationPsi,
                    min: .1,
                    max: 1,
                    divisions: 9,
                    onChanged: onTruncationChanged)),
            Text(truncationPsi.toStringAsFixed(1)),
            const SizedBox(width: 16),
          ]),
          FiltersPanel(
              initiallyExpanded: true, onFiltersCommitted: onFiltersCommitted),
        ],
      );
}

class StimuliSelectionCanvas extends StatefulWidget {
  final StimuliSelectionCubit cubit;
  final VoidCallback onResample;
  final ValueChanged<bool>? onThemeModeChanged;
  const StimuliSelectionCanvas(
      {super.key,
      required this.cubit,
      required this.onResample,
      this.onThemeModeChanged});

  @override
  State<StimuliSelectionCanvas> createState() => _StimuliSelectionCanvasState();
}

class _StimuliSelectionCanvasState extends State<StimuliSelectionCanvas> {
  final _transform = TransformationController();

  @override
  void dispose() {
    _transform.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    return ColoredBox(
      color: isDark ? AppTheme.darkCanvas : AppTheme.lightCanvas,
      child: Column(children: [
        PreviewHeaderBar(
            isDark: isDark,
            onThemeModeChanged: widget.onThemeModeChanged,
            onChangeFacePressed: widget.onResample,
            title: 'selection.preview_title'.tr(),
            actionLabel: 'selection.resample'.tr(),
            actionTooltip: 'selection.resample_tooltip'.tr()),
        Expanded(
            child: BlocBuilder<StimuliSelectionCubit, StimuliSelectionState>(
          bloc: widget.cubit,
          builder: (context, state) {
            final preview = state.preview;
            return Column(children: [
              if (state.error != null)
                Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(state.error!,
                        key: const ValueKey('selection-error'),
                        style: TextStyle(color: theme.colorScheme.error))),
              Expanded(child: LayoutBuilder(builder: (context, constraints) {
                final size = math.max(
                    24.0,
                    math.min(
                        220.0,
                        (math.min(constraints.maxWidth, constraints.maxHeight) -
                                48) /
                            3));
                return Stack(fit: StackFit.expand, children: [
                  CustomPaint(
                      painter: DottedBackgroundPainter(
                          color: isDark
                              ? const Color(0xFF3D3F43)
                              : const Color(0xFFC7CBD0),
                          spacing: 20,
                          origin: Offset.zero)),
                  InteractiveViewer(
                      transformationController: _transform,
                      minScale: .5,
                      maxScale: 6,
                      boundaryMargin: const EdgeInsets.all(1000),
                      trackpadScrollCausesScale: true,
                      scaleFactor: 160,
                      child: Center(
                          child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: List.generate(
                            3,
                            (row) => Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: List.generate(3, (column) {
                                    final index = row * 3 + column;
                                    return Padding(
                                        padding: const EdgeInsets.all(4),
                                        child: AsyncGeneratedImageTile(
                                            key: ValueKey(
                                                'selection-image-$index'),
                                            imageBytes: preview?.images[index],
                                            size: size,
                                            isLoading: state.isLoading));
                                  }),
                                )),
                      ))),
                ]);
              })),
              Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                  child: Text(
                      state.isLoading
                          ? 'selection.updating'.tr()
                          : preview == null
                              ? 'selection.empty'.tr()
                              : preview.source == 'generator'
                                  ? 'selection.generator_sample'.tr()
                                  : 'selection.ratings_sample'.tr(namedArgs: {
                                        'count': '${preview.eligibleCount}'
                                      }) +
                                      (preview.sampledWithReplacement
                                          ? 'selection.repeats'.tr()
                                          : ''),
                      textAlign: TextAlign.center)),
            ]);
          },
        )),
      ]),
    );
  }
}
