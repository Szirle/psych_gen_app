import 'package:flutter/material.dart';
import 'package:psych_gen_app/characteristic.dart';
import 'package:psych_gen_app/characteristics_selector.dart';
import 'package:psych_gen_app/custom_button.dart';
import 'package:psych_gen_app/filters.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  List<Color> colors = [Colors.teal, Colors.deepPurpleAccent, Colors.orange];
  List<Characteristic> characteristics = [
    Characteristic(characteristicName: CharacteristicName.Dominance, value: 0.5)
  ];

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
                    Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
                      Image.asset("images/logo.png", width: 40, height: 40),
                      const SizedBox(
                        width: 5,
                      ),
                      const Text(
                        "PsychGenApp",
                        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
                      )
                    ]),
                    SizedBox(
                      height: 32,
                    ),
                    Theme(
                      data: ThemeData().copyWith(dividerColor: Colors.transparent),
                      child: ExpansionTile(
                        maintainState: true,
                        title: Text(
                          'Experimental design',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        children: <Widget>[
                          Column(children: [
                            ...characteristics
                                .map(
                                  (e) => CharacteristicSelector(
                                    borderColor: colors[characteristics.indexOf(e)],
                                    onCharacteristicSelected: (characteristicName) => setState(() {
                                      e.characteristicName = characteristicName;
                                    }),
                                    onStrengthChanged: (strength) => setState(() {
                                      e.value = strength;
                                    }),
                                    onClose: () => setState(() {
                                      characteristics.remove(e);
                                    }),
                                  ),
                                )
                                .toList(),
                            CustomElevatedButton(
                              onPressed: () {
                                if (characteristics.length < 3) {
                                  characteristics.add(Characteristic(
                                      characteristicName: CharacteristicName.Conscientiousness,
                                      value: 0.5));
                                  setState(() {});
                                }
                              },
                              buttonText: 'Add variable',
                            )
                          ]),
                        ],
                        onExpansionChanged: (bool expanded) {},
                      ),
                    ),
                    Theme(
                      data: ThemeData().copyWith(dividerColor: Colors.transparent),
                      child: ExpansionTile(
                        maintainState: true,
                        title: Text(
                          'Filters',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        children: <Widget>[FiltersSection()],
                        onExpansionChanged: (bool expanded) {},
                      ),
                    ),
                    Theme(
                      data: ThemeData().copyWith(dividerColor: Colors.transparent),
                      child: ExpansionTile(
                        maintainState: true,
                        title: Text(
                          'Export',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        children: <Widget>[
                          Text('Number of images to generate for each condition'),
                          SizedBox(width: 20),
                          TextField(
                            decoration: InputDecoration(
                              border: OutlineInputBorder(),
                              labelText: '40',
                            ),
                            keyboardType: TextInputType.number,
                          ),
                          Text('A total of 1000 images will be generated.'),
                          SizedBox(height: 20),
                          ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              primary: Colors. black87, // Background color
                              onPrimary: Colors.white, // Text color
                            ),
                            onPressed: () {
                              // Implement your download dataset functionality here
                              print('Download dataset');
                            },
                            child: Text('Download dataset'),
                          ),
                        ],
                        onExpansionChanged: (bool expanded) {},
                      ),
                    ),
                  ],
                ),
              ),
            ),
            Expanded(child: Container())
          ],
        ),
      ),
    ));
  }
}

class DottedBackgroundPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1;

    var dotSize = 2.0; // Size of the dots
    var spaceBetween = 40.0; // Space between dots

    for (double i = 0; i < size.width; i += spaceBetween) {
      for (double j = 0; j < size.height; j += spaceBetween) {
        canvas.drawCircle(Offset(i, j), dotSize, paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
