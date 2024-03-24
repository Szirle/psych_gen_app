import 'package:flutter/material.dart';
import 'package:psych_gen_app/characteristic.dart';

class CharacteristicSelector extends StatefulWidget {
  final Color borderColor;
  final void Function() onClose;
  final void Function(CharacteristicName) onCharacteristicSelected;
  final void Function(double) onStrengthChanged;

  const CharacteristicSelector({
    Key? key,
    required this.borderColor,
    required this.onClose,
    required this.onCharacteristicSelected,
    required this.onStrengthChanged,
  }) : super(key: key);

  @override
  _CharacteristicSelectorState createState() => _CharacteristicSelectorState();
}

class _CharacteristicSelectorState extends State<CharacteristicSelector> {
  CharacteristicName selectedCharacteristic = CharacteristicName.Dominance;
  double strength = 0.5;

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
                offset: const Offset(0, 3), // changes position of shadow
              ),
            ],
          ),
          child: Padding(
            padding: const EdgeInsets.only(top: 8, left: 16, right: 16, bottom: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text("Variable name"),
                SizedBox(
                  height: 36,
                  child: DropdownButtonFormField<CharacteristicName>(
                    decoration: InputDecoration(
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(5.0),
                        borderSide: const BorderSide(color: Colors.black26, width: 1.0),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(5.0),
                        borderSide: const BorderSide(color: Colors.black26, width: 1.0),
                      ),
                      contentPadding: const EdgeInsets.only(top: 12, left: 12, right: 12),
                    ),
                    style: TextStyle(
                      fontFamily: 'WorkSans',
                    ),
                    value: selectedCharacteristic,
                    onChanged: (CharacteristicName? newValue) {
                      setState(() {
                        selectedCharacteristic = newValue!;
                      });
                      widget.onCharacteristicSelected(newValue!);
                    },
                    items: CharacteristicName.values
                        .map<DropdownMenuItem<CharacteristicName>>((CharacteristicName value) {
                      return DropdownMenuItem<CharacteristicName>(
                        value: value,
                        child: Text(
                          value.toString().split('.').last,
                          style: const TextStyle(
                            fontSize: 14,
                            fontFamily: 'WorkSans',
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
                SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    overlayShape: const RoundSliderOverlayShape(overlayRadius: 0.0),
                  ),
                  child: Slider(
                    activeColor: widget.borderColor,
                    inactiveColor: Colors.black12,
                    value: strength,
                    onChanged: (newRating) {
                      setState(() => strength = newRating);
                      widget.onStrengthChanged(newRating);
                    },
                    min: 0.0,
                    max: 1.0,
                  ),
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
              height: 25, // Specify the height to ensure the container is a circle
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
