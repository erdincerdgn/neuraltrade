export function generateCalendarDate(daysCount: number = 14): any[] {
    const dates: any[] = [];
    const today = new Date();
    
    const weekDays = [
        'Sunday',
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday'
    ];

    const months = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December'
    ];

    for ( let i = 0; i < daysCount; i++ ) {
        const date = new Date(today);
        date.setDate(today.getDate() + 1);

        dates.push({
            day: date.getDate().toString(),
            weekDay: weekDays[date.getDay()],
            month: months[date.getMonth()],
        });
    }

    return dates;
}