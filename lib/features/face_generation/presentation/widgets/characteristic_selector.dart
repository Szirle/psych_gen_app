import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/distribution_range_selector.dart';

class CharacteristicSelector extends StatefulWidget {
  final Color borderColor;
  final void Function() onClose;
  final void Function(ManipulatedDimensionName) onCharacteristicSelected;
  final void Function(double) onStrengthChanged;
  final void Function(int) onNLevelChanged;
  final void Function(double, double) onRangeChanged;
  final ManipulatedDimension manipulatedDimension;
  final List<ManipulatedDimension> allManipulatedDimensions;

  CharacteristicSelector({
    Key? key,
    required this.borderColor,
    required this.onClose,
    required this.onCharacteristicSelected,
    required this.onStrengthChanged,
    required this.onNLevelChanged,
    required this.onRangeChanged,
    required this.manipulatedDimension,
    required this.allManipulatedDimensions,
  }) : super(key: key);

  @override
  _CharacteristicSelectorState createState() => _CharacteristicSelectorState();
}

class _CharacteristicSelectorState extends State<CharacteristicSelector> {
  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Container(
          margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 24),
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border.all(color: widget.borderColor, width: 2),
            borderRadius: BorderRadius.circular(5),
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.5),
                spreadRadius: 1,
                blurRadius: 5,
                offset: const Offset(0, 3),
              ),
            ],
          ),
          child: Padding(
            padding:
                const EdgeInsets.only(top: 8, left: 16, right: 16, bottom: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text('characteristic.variable_name'.tr()),
                SizedBox(
                  height: 36,
                  child: DropdownButtonFormField<ManipulatedDimensionName>(
                    decoration: InputDecoration(
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(5.0),
                        borderSide:
                            const BorderSide(color: Colors.black26, width: 1.0),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(5.0),
                        borderSide:
                            const BorderSide(color: Colors.black26, width: 1.0),
                      ),
                      contentPadding:
                          const EdgeInsets.only(top: 12, left: 12, right: 12),
                    ),
                    style: const TextStyle(
                      fontFamily: 'WorkSans',
                    ),
                    value: widget.manipulatedDimension.name,
                    onChanged: (ManipulatedDimensionName? newValue) {
                      setState(() {
                        widget.manipulatedDimension.name = newValue!;
                      });
                      widget.onCharacteristicSelected(newValue!);
                    },
                    items: ManipulatedDimensionName.values
                        .map<DropdownMenuItem<ManipulatedDimensionName>>(
                            (ManipulatedDimensionName value) {
                      final bool isSelected = widget.allManipulatedDimensions
                          .any((dim) =>
                              dim.name == value &&
                              dim != widget.manipulatedDimension);
                      return DropdownMenuItem<ManipulatedDimensionName>(
                        value: value,
                        enabled: !isSelected,
                        child: Text(
                          value.toString().split('.').last,
                          style: TextStyle(
                            fontSize: 14,
                            fontFamily: 'WorkSans',
                            color: isSelected ? Colors.grey : Colors.black,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
                const SizedBox(height: 5),
                Tooltip(
                  message: 'tooltip.variable_strength'.tr(),
                  child: Text('characteristic.variable_strength'.tr()),
                ),
                Row(
                  children: [
                    Expanded(
                      child: SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          overlayShape:
                              const RoundSliderOverlayShape(overlayRadius: 0.0),
                        ),
                        child: Slider(
                          activeColor: widget.borderColor,
                          inactiveColor: Colors.black12,
                          value: widget.manipulatedDimension.strength,
                          label: widget.manipulatedDimension.strength
                              .toStringAsFixed(1),
                          onChanged: (newRating) {
                            setState(() => widget
                                .manipulatedDimension.strength = newRating);
                            widget.onStrengthChanged(newRating);
                          },
                          min: 1.0,
                          max: 50.0,
                        ),
                      ),
                    ),
                    Container(
                      width: 50,
                      alignment: Alignment.center,
                      child: Text(
                        widget.manipulatedDimension.strength.toStringAsFixed(1),
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 5),
                Tooltip(
                  message: 'tooltip.number_of_levels'.tr(),
                  child: Text('characteristic.number_of_levels'.tr()),
                ),
                Row(
                  children: [
                    Expanded(
                      child: SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          overlayShape:
                              const RoundSliderOverlayShape(overlayRadius: 0.0),
                        ),
                        child: Slider(
                          activeColor: widget.borderColor,
                          inactiveColor: Colors.black12,
                          value: widget.manipulatedDimension.nLevels.toDouble(),
                          min: 1.0,
                          max: 5.0,
                          divisions: 2,
                          onChanged: (newValue) {
                            setState(() => widget.manipulatedDimension.nLevels =
                                newValue.round());
                            widget.onNLevelChanged(
                                widget.manipulatedDimension.nLevels);
                          },
                        ),
                      ),
                    ),
                    Container(
                      width: 50,
                      alignment: Alignment.center,
                      child: Text(
                        widget.manipulatedDimension.nLevels.toString(),
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        Positioned(
          right: 0,
          top: 16,
          child: InkWell(
            onTap: () {
              widget.onClose();
            },
            child: Container(
              width: 25,
              height: 25,
              decoration: BoxDecoration(
                color: widget.borderColor,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.close,
                color: Colors.white,
                size: 20,
              ),
            ),
          ),
        ),
      ],
    );
  }
}
