import 'package:flutter/material.dart';
import 'package:psych_gen_app/model/face_manipulation_request.dart';
import 'package:psych_gen_app/model/manipulated_dimension.dart';
import 'package:psych_gen_app/model/manipulated_dimension_name.dart';

class CharacteristicSelector extends StatefulWidget {
  final Color borderColor;
  final void Function() onClose;
  final void Function(ManipulatedDimensionName) onCharacteristicSelected;
  final void Function(double) onStrengthChanged;
  final void Function(int)
      onNLevelChanged; // Callback for the odd integer slider
  ManipulatedDimension manipulatedDimension;
  final List<ManipulatedDimension> allManipulatedDimensions;

  CharacteristicSelector({
    Key? key,
    required this.borderColor,
    required this.onClose,
    required this.onCharacteristicSelected,
    required this.onStrengthChanged,
    required this.onNLevelChanged, // New callback for odd levels slider
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
          margin: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border.all(color: widget.borderColor, width: 2),
            borderRadius: BorderRadius.circular(5),
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.5),
                spreadRadius: 1,
                blurRadius: 5,
                offset: const Offset(0, 3), // Changes position of shadow
              ),
            ],
          ),
          child: Padding(
            padding:
                const EdgeInsets.only(top: 8, left: 16, right: 16, bottom: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text("Variable name"),
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
                const SizedBox(
                  height: 5,
                ),
                const Text("Variable strength"),
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
                        widget.manipulatedDimension.strength
                            .toStringAsFixed(1), // Display strength
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ],
                ),
                const SizedBox(
                  height: 5,
                ),
                const Text("Number of levels"),
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
                          max: 11.0,
                          divisions: 5,
                          onChanged: (newValue) {
                            setState(() => widget.manipulatedDimension.nLevels =
                                newValue.round());
                            widget.onNLevelChanged(widget.manipulatedDimension
                                .nLevels); // Callback for odd levels
                          },
                        ),
                      ),
                    ),
                    Container(
                      width: 50,
                      alignment: Alignment.center,
                      child: Text(
                        widget.manipulatedDimension.nLevels
                            .toString(), // Display the current odd level
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ],
                ),
                const SizedBox(),
              ],
            ),
          ),
        ),
        Positioned(
          right: 0,
          top: 0,
          child: InkWell(
            onTap: () {
              widget.onClose(); // Invoke the close callback
            },
            child: Container(
              width: 25, // Specify the width
              height:
                  25, // Specify the height to ensure the container is a circle
              decoration: BoxDecoration(
                color: widget.borderColor,
                shape: BoxShape.circle, // This makes the container a circle
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
