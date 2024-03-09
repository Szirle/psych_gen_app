import 'package:flutter/material.dart';
import 'package:psych_gen_app/characteristic.dart';

class CharacteristicSelector extends StatefulWidget {
  final Color borderColor;
  final void Function() onClose;
  final void Function(CharacteristicName) onCharacteristicSelected;
  final void Function(double) onStrengthChanged;

  CharacteristicSelector({
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
                DropdownButtonFormField<CharacteristicName>(
                  decoration: InputDecoration(
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide: const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide: const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    contentPadding: const EdgeInsets.symmetric(vertical: 2.0, horizontal: 10.0),
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
                        // Make sure the text style fits within the new padding if necessary
                        style: const TextStyle(fontSize: 14), // Adjust font size as necessary
                      ),
                    );
                  }).toList(),
                ),
                const SizedBox(
                  height: 5,
                ),
                const Text("Variable strength"),
                SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    overlayShape: RoundSliderOverlayShape(overlayRadius: 0.0),
                  ),
                  child: Slider(
                    activeColor: widget.borderColor,
                    inactiveColor: Colors.black26,
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
              decoration: BoxDecoration(
                color: widget.borderColor,
                borderRadius: const BorderRadius.all(
                  Radius.circular(5),
                ),
              ),
              child: const Icon(Icons.close, color: Colors.white),
            ),
          ),
        ),
      ],
    );
  }
}
