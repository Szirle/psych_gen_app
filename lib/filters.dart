import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:psych_gen_app/custom_button.dart';

class FiltersSection extends StatefulWidget {
  @override
  _FiltersSectionState createState() => _FiltersSectionState();
}

class _FiltersSectionState extends State<FiltersSection> {
  List<String> mustHaveFilters = ['Skin color: white', 'Hair: bald'];
  List<String> cantHaveFilters = ['Hair: beard'];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 350,
        ),
        Text('Must have:', style: Theme.of(context).textTheme.subtitle1),
        Wrap(
          spacing: 8.0, // Gap between adjacent chips
          runSpacing: 4.0, // Gap between lines
          children: mustHaveFilters
              .map((filter) => Chip(
                    backgroundColor: Colors.grey.shade300,
                    label: Text(filter),
                    onDeleted: () {
                      setState(() {
                        mustHaveFilters.remove(filter);
                      });
                    },
                  ))
              .toList(),
        ),
        SizedBox(height: 10),
        CustomElevatedButton(
          onPressed: () {},
          buttonText: 'Add filter',
        ),
        SizedBox(height: 20),
        Text('Can’t have:', style: Theme.of(context).textTheme.subtitle1),
        Wrap(
          spacing: 8.0,
          runSpacing: 4.0,
          children: cantHaveFilters
              .map((filter) => Chip(
                    backgroundColor: Colors.grey.shade300,
                    label: Text(filter),
                    onDeleted: () {
                      setState(() {
                        cantHaveFilters.remove(filter);
                      });
                    },
                  ))
              .toList(),
        ),
        SizedBox(height: 10),
        CustomElevatedButton(
          onPressed: () {},
          buttonText: 'Add filter',
        ),
      ],
    );
  }
}
