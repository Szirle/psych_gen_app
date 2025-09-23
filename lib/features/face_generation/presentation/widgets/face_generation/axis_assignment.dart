import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/characteristic_selector.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/face_generation/preview_painters.dart';

typedef AxisValueSetter = void Function(String axis, ManipulatedDimension? dim);

class AxisAssignment extends StatelessWidget {
  final List<ManipulatedDimension> manipulatedDimensions;
  final Map<ManipulatedDimension, Color> dimensionColors;
  final ManipulatedDimension? xAxisDim;
  final ManipulatedDimension? yAxisDim;
  final ManipulatedDimension? sliderDim;
  final AxisValueSetter onAxisSet;
  final VoidCallback onDimsChanged;

  const AxisAssignment({
    super.key,
    required this.manipulatedDimensions,
    required this.dimensionColors,
    required this.xAxisDim,
    required this.yAxisDim,
    required this.sliderDim,
    required this.onAxisSet,
    required this.onDimsChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 8.0, left: 8.0, right: 8.0),
          child: Column(
            children: List.generate(manipulatedDimensions.length, (index) {
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
          children: manipulatedDimensions.asMap().entries.map((entry) {
            final index = entry.key;
            final dim = entry.value;
            return Padding(
              key: ValueKey(dim),
              padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
              child: Stack(
                children: [
                  CharacteristicSelector(
                    manipulatedDimension: dim,
                    allManipulatedDimensions: manipulatedDimensions,
                    borderColor: dimensionColors[dim] ?? Colors.grey,
                    onRangeChanged: (start, end) {
                      dim.rangeStart = start;
                      dim.rangeEnd = end;
                      onDimsChanged();
                    },
                    onCharacteristicSelected: (characteristicName) {
                      final isAlreadySelected = manipulatedDimensions
                          .any((d) => d != dim && d.name == characteristicName);
                      if (isAlreadySelected) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content:
                                Text('${characteristicName.name} is already selected.'),
                            backgroundColor: Colors.red,
                          ),
                        );
                      } else {
                        dim.name = characteristicName;
                        onDimsChanged();
                      }
                    },
                    onStrengthChanged: (strength) {
                      dim.strength = strength;
                      onDimsChanged();
                    },
                    onClose: () {
                      manipulatedDimensions.remove(dim);
                      onDimsChanged();
                    },
                    onNLevelChanged: (nLevel) {
                      dim.nLevels = nLevel;
                      onDimsChanged();
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
            if (oldIndex < newIndex) {
              newIndex -= 1;
            }
            final item = manipulatedDimensions.removeAt(oldIndex);
            manipulatedDimensions.insert(newIndex, item);
            onDimsChanged();
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
}


