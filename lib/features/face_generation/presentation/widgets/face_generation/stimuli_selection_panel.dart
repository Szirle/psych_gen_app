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
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Section header
        const Padding(
          padding: EdgeInsets.only(left: 4, bottom: 16),
          child: Text(
            'Sampling distribution',
            style: TextStyle(
              fontFamily: 'WorkSans',
              fontWeight: FontWeight.w600,
              fontSize: 13,
              letterSpacing: 0.5,
            ),
          ),
        ),

        // Description info card
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: scheme.outlineVariant.withValues(alpha: 0.5),
            ),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.only(top: 1, right: 10),
                child: Icon(
                  Icons.info_outline_rounded,
                  size: 16,
                  color: scheme.onSurfaceVariant,
                ),
              ),
              Expanded(
                child: Text(
                  'selection.description'.tr(),
                  style: TextStyle(
                    fontFamily: 'WorkSans',
                    fontSize: 12.5,
                    height: 1.5,
                    color: scheme.onSurfaceVariant,
                  ),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 20),

        // Truncation Psi
        Padding(
          padding: const EdgeInsets.only(left: 4),
          child: Text(
            'settings.truncation_psi'.tr(),
            style: TextStyle(
              fontFamily: 'WorkSans',
              fontSize: 13,
              fontWeight: FontWeight.w500,
              color: scheme.onSurface,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Row(children: [
          Expanded(
            child: Slider(
              value: truncationPsi,
              min: .1,
              max: 1,
              divisions: 9,
              onChanged: onTruncationChanged,
            ),
          ),
          Container(
            width: 40,
            alignment: Alignment.center,
            child: Text(
              truncationPsi.toStringAsFixed(1),
              style: TextStyle(
                fontFamily: 'WorkSans',
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: scheme.onSurface,
              ),
            ),
          ),
          const SizedBox(width: 8),
        ]),

        const SizedBox(height: 8),

        // Filters
        FiltersPanel(
          initiallyExpanded: true,
          onFiltersCommitted: onFiltersCommitted,
        ),
      ],
    );
  }
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
    final scheme = theme.colorScheme;
    final isDark = theme.brightness == Brightness.dark;
    final dotColor =
        isDark ? const Color(0xFF3D3F43) : const Color(0xFFC7CBD0);

    // Stack layout: canvas extends under the header bar, matching
    // the manipulation tab's layout.
    return ColoredBox(
      color: isDark ? AppTheme.darkCanvas : AppTheme.lightCanvas,
      child: Stack(
        children: [
          // Full-bleed canvas with interactive viewer
          Positioned.fill(
            child: BlocBuilder<StimuliSelectionCubit, StimuliSelectionState>(
              bloc: widget.cubit,
              builder: (context, state) {
                final preview = state.preview;
                return LayoutBuilder(builder: (context, constraints) {
                  final tileSize = math.max(
                      24.0,
                      math.min(
                          220.0,
                          (math.min(constraints.maxWidth,
                                      constraints.maxHeight) -
                                  48) /
                              3));
                  final gridExtent = tileSize * 3 + 8 * 3;
                  final dotSpacing = tileSize / 3;

                  return MouseRegion(
                    cursor: SystemMouseCursors.grab,
                    child: InteractiveViewer(
                      transformationController: _transform,
                      minScale: 0.5,
                      maxScale: 6,
                      boundaryMargin: const EdgeInsets.all(1000),
                      trackpadScrollCausesScale: true,
                      scaleFactor: 160,
                      clipBehavior: Clip.hardEdge,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          // Dots inside InteractiveViewer so they pan/zoom
                          CustomPaint(
                            painter: DottedBackgroundPainter(
                              color: dotColor,
                              spacing: dotSpacing,
                              origin: Offset(
                                (constraints.maxWidth - gridExtent) / 2,
                                (constraints.maxHeight - gridExtent) / 2,
                              ),
                            ),
                          ),
                          Center(
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
                                        size: tileSize,
                                        isLoading: state.isLoading,
                                      ),
                                    );
                                  }),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                });
              },
            ),
          ),

          // Header bar overlaying the canvas (like manipulation tab)
          Positioned(
            left: 0,
            top: 0,
            right: 0,
            child: PreviewHeaderBar(
              isDark: isDark,
              onThemeModeChanged: widget.onThemeModeChanged,
              onChangeFacePressed: widget.onResample,
              title: 'selection.preview_title'.tr(),
              actionLabel: 'selection.resample'.tr(),
              actionTooltip: 'selection.resample_tooltip'.tr(),
            ),
          ),

          // Status chip at bottom + error
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: BlocBuilder<StimuliSelectionCubit, StimuliSelectionState>(
              bloc: widget.cubit,
              builder: (context, state) {
                final preview = state.preview;
                return Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (state.error != null)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 10),
                          decoration: BoxDecoration(
                            color: scheme.errorContainer,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.warning_amber_rounded,
                                  size: 16, color: scheme.error),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(state.error!,
                                    key: const ValueKey('selection-error'),
                                    style: TextStyle(
                                      color: scheme.onErrorContainer,
                                      fontSize: 12.5,
                                      fontFamily: 'WorkSans',
                                    )),
                              ),
                            ],
                          ),
                        ),
                      ),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                      child: _buildStatusChip(context, state, preview, isDark),
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusChip(BuildContext context, StimuliSelectionState state,
      dynamic preview, bool isDark) {
    final scheme = Theme.of(context).colorScheme;

    final String text;
    final IconData icon;

    if (state.isLoading) {
      text = 'selection.updating'.tr();
      icon = Icons.hourglass_top_rounded;
    } else if (preview == null) {
      text = 'selection.empty'.tr();
      icon = Icons.tune_rounded;
    } else if (preview.source == 'generator') {
      text = 'selection.generator_sample'.tr();
      icon = Icons.auto_awesome_rounded;
    } else {
      text = 'selection.ratings_sample'.tr(namedArgs: {
            'count': '${preview.eligibleCount}'
          }) +
          (preview.sampledWithReplacement ? 'selection.repeats'.tr() : '');
      icon = Icons.people_outline_rounded;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: isDark
            ? Colors.white.withValues(alpha: 0.06)
            : Colors.black.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: scheme.onSurfaceVariant),
          const SizedBox(width: 8),
          Flexible(
            child: Text(text,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontFamily: 'WorkSans',
                  fontSize: 11.5,
                  color: scheme.onSurfaceVariant,
                )),
          ),
        ],
      ),
    );
  }
}
